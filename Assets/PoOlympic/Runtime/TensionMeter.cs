using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// One number for "how exciting is this moment" (0..1), shared by the camera director (feature 2), the stadium
    /// audio (feature 10), the balance overlay (feature 6), the stats card (feature 7) and the commentary. Built from the
    /// event board and every athlete's telemetry:
    ///   danger per athlete  = max(1 − brain confidence, balance danger (smoothed CoM margin to the support polygon,
    ///                         stationary events, capped at 0.6), torso tilt towards the 60° fall line, tilt rate)
    ///   closeness (races)   = how near 2nd is to the leader along the race axis
    ///   tension             = phase base + 0.45·max danger + 0.3·closeness + 0.15·share of athletes out; fast attack,
    ///                         slow release; a spike at the result
    /// Hot athlete = the one in most danger (if it is real danger), else the leader. Raises NearFall / Save / Fall events.
    /// </summary>
    [DefaultExecutionOrder(150)]
    public class TensionMeter : MonoBehaviour
    {
        public static TensionMeter Instance { get; private set; }

        public MonoBehaviour board;                    // IBroadcastBoard
        public BroadcastDirector.Kind kind = BroadcastDirector.Kind.Race;
        public Vector3 forward = Vector3.right;
        [Tooltip("Danger above this = a near fall (camera cut, gasp, overlay).")]
        public float nearFallDanger = 0.7f;
        [Tooltip("Back below this after a near fall without going out = a save.")]
        public float saveDanger = 0.35f;
        [Tooltip("Danger must stay above nearFallDanger this long (s) to count as a near fall.")]
        public float nearFallHold = 0.3f;
        [Tooltip("Minimum seconds between two near-fall calls, and between two save calls (all athletes).")]
        public float callCooldown = 4f;

        public float Tension { get; private set; }
        public PolicyRunner Hot { get; private set; }
        public float HotDanger { get; private set; }
        public PolicyRunner Leader { get; private set; }
        public float LeadGapM { get; private set; } = float.NaN;
        public readonly Dictionary<PolicyRunner, float> Danger = new();

        public event Action<PolicyRunner> NearFall, Save, Fall, HeatStarted;
        public event Action<BoardPhase> PhaseChanged;

        IBroadcastBoard B => board as IBroadcastBoard;
        readonly Dictionary<PolicyRunner, AthleteTelemetry> _tel = new();
        readonly Dictionary<PolicyRunner, float> _nearSince = new(), _overSince = new();
        float _lastNearCall = -99f, _lastSaveCall = -99f;
        readonly HashSet<PolicyRunner> _out = new();
        BoardPhase _phase = BoardPhase.Ready;
        int _heat = -1;
        float _resultAt;

        void OnEnable() => Instance = this;
        void OnDisable() { if (Instance == this) Instance = null; }

        public AthleteTelemetry TelemetryOf(PolicyRunner r)
        {
            if (r == null) return null;
            if (_tel.TryGetValue(r, out var t) && t != null) return t;
            return _tel[r] = r.GetComponent<AthleteTelemetry>();
        }

        void Update()
        {
            var b = B;
            if (b == null) return;
            var rows = b.Rows.ToList();
            if (b.Heat != _heat)
            {
                _heat = b.Heat;
                _out.Clear();
                _nearSince.Clear();
                _overSince.Clear();
                foreach (var r in rows) TelemetryOf(r.runner)?.ResetHeat();
                HeatStarted?.Invoke(null);
            }
            if (b.BoardState != _phase)
            {
                _phase = b.BoardState;
                if (_phase == BoardPhase.Result) _resultAt = Time.unscaledTime;
                PhaseChanged?.Invoke(_phase);
            }

            float maxDanger = 0f;
            PolicyRunner hot = null;
            int live = 0, outCount = 0;
            Danger.Clear();
            foreach (var row in rows)
            {
                var r = row.runner;
                if (r == null || !r.isActiveAndEnabled) continue;
                if (row.bad)
                {
                    outCount++;
                    if (_out.Add(r) && _phase == BoardPhase.Live) Fall?.Invoke(r);
                    continue;
                }
                live++;
                float d = DangerOf(TelemetryOf(r));
                Danger[r] = d;
                if (d > maxDanger) { maxDanger = d; hot = r; }
                // near fall → save bookkeeping (live phase only)
                if (_phase != BoardPhase.Live) continue;
                // a near fall = danger held for nearFallHold s (not a single wobble frame); a save = back to safety
                // within 6 s without going out. Calls are rate-limited so a wobbly heat does not flood the ticker.
                float now = Time.unscaledTime;
                if (d >= nearFallDanger)
                {
                    if (!_overSince.ContainsKey(r)) _overSince[r] = now;
                    if (!_nearSince.ContainsKey(r) && now - _overSince[r] >= nearFallHold)
                    {
                        _nearSince[r] = now;
                        if (now - _lastNearCall >= callCooldown) { _lastNearCall = now; NearFall?.Invoke(r); }
                    }
                }
                else
                {
                    _overSince.Remove(r);
                    if (d <= saveDanger && _nearSince.TryGetValue(r, out var since))
                    {
                        _nearSince.Remove(r);
                        if (now - since < 6f && now - _lastSaveCall >= callCooldown) { _lastSaveCall = now; Save?.Invoke(r); }
                    }
                }
            }

            // leader + closeness (races: progress along the race axis)
            Leader = rows.FirstOrDefault(x => !x.bad && x.runner != null && x.runner.isActiveAndEnabled).runner;
            float closeness = 0f;
            LeadGapM = float.NaN;
            if (kind == BroadcastDirector.Kind.Race && live >= 2)
            {
                var f = new Vector3(forward.x, 0, forward.z).normalized;
                var progress = rows.Where(x => !x.bad && x.runner != null && x.runner.isActiveAndEnabled)
                                   .Select(x => TelemetryOf(x.runner)).Where(t => t != null && t.Ready)
                                   .Select(t => Vector3.Dot(t.PelvisPosition, f)).OrderByDescending(p => p).ToList();
                if (progress.Count >= 2)
                {
                    LeadGapM = progress[0] - progress[1];
                    closeness = Mathf.Clamp01(1f - LeadGapM / 2f);
                }
            }

            HotDanger = maxDanger;
            Hot = maxDanger >= 0.45f ? hot : Leader;
            float target = _phase switch
            {
                BoardPhase.Ready => 0.15f,
                BoardPhase.Result => Mathf.Lerp(1f, 0.35f, Mathf.Clamp01((Time.unscaledTime - _resultAt) / 6f)),
                _ => Mathf.Clamp01(0.3f + 0.45f * maxDanger + 0.3f * closeness + 0.15f * (rows.Count > 0 ? (float)outCount / rows.Count : 0f)),
            };
            float rate = target > Tension ? 4f : 0.6f;          // fast attack, slow release
            Tension += (target - Tension) * (1f - Mathf.Exp(-rate * Time.unscaledDeltaTime));
        }

        /// <summary>0 = rock solid, 1 = going down.</summary>
        public float DangerOf(AthleteTelemetry t)
        {
            if (t == null || !t.Ready) return 0f;
            float conf = float.IsNaN(t.Confidence) ? 0f : 1f - t.Confidence;
            bool crawler = t.Runner.crawlSteering;           // on all fours the torso is horizontal by design
            float tilt = crawler ? 0f : Mathf.InverseLerp(20f, 55f, t.TiltDeg);
            float tiltRate = crawler ? 0f : Mathf.InverseLerp(60f, 240f, Mathf.Abs(t.TiltRateDps));
            // stationary events: CoM against the support polygon (smoothed — a recovery step briefly stands on one foot,
            // which is outside static balance but not a fall), capped so a cut / near-fall call needs the brain or the
            // torso to agree
            float balance = 0f;
            if (kind == BroadcastDirector.Kind.Arena && !float.IsNaN(t.BalanceMarginSmooth))
                balance = 0.6f * Mathf.InverseLerp(0.08f, -0.05f, t.BalanceMarginSmooth);   // 8 cm inside = safe, 5 cm outside = gone
            return Mathf.Clamp01(Mathf.Max(Mathf.Max(conf, balance), Mathf.Max(tilt, tiltRate * 0.8f)));
        }
    }
}
