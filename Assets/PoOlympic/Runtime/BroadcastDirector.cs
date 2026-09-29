using System.Collections.Generic;
using System.Linq;
using Mujoco;
using Unity.Cinemachine;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Broadcast camera director (D3, feature 2: tension-driven Cinemachine cuts). Framing targets:
    ///   Race  — focus = the leader along the race axis, centred across the lanes
    ///   Arena — focus = centre of the athletes (stationary events)
    ///   Hot   — the athlete in most danger (TensionMeter), Winner — place 1 at the result
    /// The director only moves three proxy transforms (focus / hot / winner) and chooses which Cinemachine camera is
    /// live (priority); Cinemachine does the following, damping, blends, hand-held noise and impact shake (impulse):
    ///   Ready   establishing (the authored BroadcastCamera offset)
    ///   Live    races cycle trackside / head-on / high-wide (faster cuts as tension rises); arenas orbit slowly;
    ///           a real near fall (HotDanger ≥ hotCutDanger) eases to a hand-held close-up of that athlete and holds
    ///           it until the danger is over (or the athlete is out)
    ///   Result  eases to a front three-quarter close-up of the winner
    /// Scenes without the Cinemachine rig (wide == null) keep the original direct-transform framing.
    /// Letterboxing stays with BroadcastCamera (its target is cleared, so it only letterboxes).
    /// </summary>
    [RequireComponent(typeof(Camera))]
    [DefaultExecutionOrder(-50)]      // before CinemachineBrain (order 0) reads the proxies
    public class BroadcastDirector : MonoBehaviour
    {
        public enum Kind { Race, Arena }
        public enum Shot { Establishing, Trackside, HeadOn, HighWide, Hot, Winner }
        public MonoBehaviour board;                     // IBroadcastBoard
        public Kind kind = Kind.Race;
        [Tooltip("Race direction in Unity world space (MuJoCo +x = Unity +x).")]
        public Vector3 forward = Vector3.right;
        [Tooltip("Establishing shot: camera offset from the focus in world space (as authored by the scene builder).")]
        public Vector3 establishing = new(-3.5f, 3.2f, -13f);
        public float lookHeight = 0.9f;
        public float shotSeconds = 5f;
        public float followSharpness = 3f;
        public float orbitSpeed = 0.12f;                // rad/s (arena)
        [Tooltip("Cap on the camera height (m): Event 23 keeps every shot under the trench ceiling.")]
        public float maxCameraHeight = 100f;

        [Header("Cinemachine rig (PoOlympic › Broadcast › Upgrade broadcast FX)")]
        public CinemachineCamera wide;
        public CinemachineCamera headOn, high, hot, winner;
        public Transform focusProxy, hotProxy, winnerProxy;
        public TensionMeter tension;
        [Tooltip("Danger (TensionMeter) at which the director cuts to the athlete in trouble.")]
        public float hotCutDanger = 0.75f;
        public float minShotSeconds = 2.2f;
        [Tooltip("Seconds after a hot close-up before the next one (unless the danger is extreme, ≥ 0.9).")]
        public float hotCooldown = 5f;

        public Shot Current => _shotNow;

        IBroadcastBoard B => board as IBroadcastBoard;
        readonly Dictionary<PolicyRunner, Transform> _pelvis = new();
        float _shotClock;
        int _shot;
        BoardPhase _last = BoardPhase.Ready;
        float _orbit;
        Vector3 _pos, _look;
        bool _init, _cut;
        Shot _shotNow = Shot.Establishing;
        float _shotHeld, _hotClock, _hotLeft = -99f;
        PolicyRunner _hotRunner;
        CinemachineBrain _brain;

        void Start()
        {
            var cam = GetComponent<BroadcastCamera>();
            if (cam != null) cam.target = null;         // this director owns the framing
            _brain = GetComponent<CinemachineBrain>();
            if (tension == null) tension = FindAnyObjectByType<TensionMeter>();
        }

        Transform Pelvis(PolicyRunner r)
        {
            if (r == null) return null;
            if (_pelvis.TryGetValue(r, out var t) && t != null) return t;
            var body = FindObjectsByType<MjBody>(FindObjectsSortMode.None).FirstOrDefault(b => b.name == r.athletePrefix + "pelvis");
            _pelvis[r] = body != null ? body.transform : null;
            return _pelvis[r];
        }

        void LateUpdate()
        {
            var b = B;
            if (b == null) return;
            var rows = b.Rows.ToList();
            var bodies = rows.Select(r => (r, t: Pelvis(r.runner))).Where(x => x.t != null && x.r.runner.gameObject.activeInHierarchy).ToList();
            if (bodies.Count == 0) return;
            var f = new Vector3(forward.x, 0, forward.z).normalized;
            var left = Vector3.Cross(Vector3.up, f);    // Unity is left-handed: up × forward = left

            var centre = bodies.Aggregate(Vector3.zero, (s, x) => s + x.t.position) / bodies.Count;
            Vector3 focus;
            if (kind == Kind.Race)
            {
                var lead = bodies.FirstOrDefault(x => !x.r.bad);
                var leadPos = (lead.t != null ? lead.t : bodies[0].t).position;
                // along the race axis: the leader, a little ahead; across: the lane centre (constant)
                focus = centre + f * (Vector3.Dot(leadPos - centre, f) + 1.0f);
            }
            else focus = centre;
            focus.y = lookHeight;

            bool phaseChanged = b.BoardState != _last;
            if (phaseChanged) { _shotClock = 0; _shot = 0; _last = b.BoardState; _snap = true; }
            _shotClock += Time.unscaledDeltaTime;
            if (wide != null) Direct(b, bodies, focus, f, left);
            else Legacy(b, bodies, focus, f, left);
        }

        // ---------------------------------------------------------------- Cinemachine
        void Direct(IBroadcastBoard b, List<((int place, string name, string result, bool bad, PolicyRunner runner) r, Transform t)> bodies,
                    Vector3 focus, Vector3 f, Vector3 left)
        {
            float k = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
            // a new phase (new heat at the start line, the gun, the result) is a hard cut: no drifting back along the track
            bool snap = !_init || (_snap && b.BoardState != BoardPhase.Result);
            focusProxy.position = snap ? focus : Vector3.Lerp(focusProxy.position, focus, k);
            if (snap) foreach (var c in new[] { wide, headOn, high, hot, winner }) if (c != null) c.PreviousStateIsValid = false;
            _snap = false;
            _init = true;
            float t = tension != null ? tension.Tension : 0.3f;
            _shotHeld += Time.unscaledDeltaTime;

            // who is the story right now
            var hotRunner = tension != null && tension.HotDanger >= hotCutDanger ? tension.Hot : null;
            var w = bodies.FirstOrDefault(x => x.r.place == 1);
            if (w.t != null) winnerProxy.position = w.t.position + Vector3.up * 0.2f;
            Shot want;
            switch (b.BoardState)
            {
                case BoardPhase.Result:
                    want = Shot.Winner;
                    break;
                case BoardPhase.Live:
                    bool hotOk = _hotRunner != null && bodies.Any(x => x.r.runner == _hotRunner && !x.r.bad);
                    if (_shotNow == Shot.Hot && hotOk && (_hotClock < minShotSeconds ||
                        (tension != null && tension.Danger.TryGetValue(_hotRunner, out var dz) && dz > 0.35f && _hotClock < 6f)))
                    {
                        want = Shot.Hot;                                     // stay with the athlete in trouble
                    }
                    else if (hotRunner != null && _shotHeld >= minShotSeconds * 0.5f && _shotNow != Shot.Hot &&
                             (Time.unscaledTime - _hotLeft >= hotCooldown || tension.HotDanger >= 0.9f))
                    {
                        want = Shot.Hot;
                        _hotRunner = hotRunner;
                        _hotClock = 0f;
                    }
                    else if (kind == Kind.Race)
                    {
                        float len = shotSeconds * Mathf.Lerp(1.1f, 0.55f, t);     // tension → faster cutting
                        if (_shotNow == Shot.Hot || _shotClock > len) { _shotClock = 0; _shot = (_shot + 1) % 3; }
                        want = _shot switch { 1 => Shot.HeadOn, 2 => Shot.HighWide, _ => Shot.Trackside };
                    }
                    else want = Shot.Trackside;
                    break;
                default:
                    want = Shot.Establishing;
                    _orbit = 0;
                    break;
            }
            if (want == Shot.Hot)
            {
                _hotClock += Time.unscaledDeltaTime;
                var ht = Pelvis(_hotRunner);
                if (ht != null) hotProxy.position = Vector3.Lerp(hotProxy.position, ht.position, _shotNow == Shot.Hot ? k * 3f : 1f);
            }

            // offsets (world space, relative to the proxies)
            if (kind == Kind.Arena && b.BoardState == BoardPhase.Live)
            {
                if (_shotClock > shotSeconds * 1.5f) _orbit += orbitSpeed * Time.unscaledDeltaTime;
                SetOffset(wide, focusProxy, Quaternion.AngleAxis(Mathf.Rad2Deg * _orbit, Vector3.up) * establishing);
            }
            else SetOffset(wide, focusProxy, establishing);
            SetOffset(headOn, focusProxy, f * 9f + Vector3.up * 1.6f - left * 2.0f);
            SetOffset(high, focusProxy, -f * 11f + Vector3.up * 7f - left * 6f);
            bool low = _hotRunner != null && _hotRunner.crawlSteering;
            SetOffset(hot, hotProxy, f * 2.6f - left * 4.6f + Vector3.up * (low ? 0.6f : 1.1f));   // full body, 5.3 m
            SetOffset(winner, winnerProxy, f * 5.5f - left * 3.2f + Vector3.up * 1.2f);

            if (want != _shotNow || !_liveSet) Activate(want);
        }

        bool _liveSet, _snap;

        void Activate(Shot s)
        {
            var from = _shotNow;
            if (from == Shot.Hot) _hotLeft = Time.unscaledTime;
            _shotNow = s;
            _shotHeld = 0f;
            _liveSet = true;
            if (_brain != null)
            {
                _brain.DefaultBlend = s switch
                {
                    Shot.Hot => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, 0.45f),
                    Shot.Winner => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, 1.1f),
                    _ when from == Shot.Hot => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, 0.7f),
                    _ => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.Cut, 0f),   // broadcast cuts
                };
            }
            var live = CameraOf(s);
            foreach (var c in new[] { wide, headOn, high, hot, winner })
                if (c != null) c.Priority = c == live ? 20 : 10;
        }

        CinemachineCamera CameraOf(Shot s) => s switch
        {
            Shot.HeadOn => headOn,
            Shot.HighWide => high,
            Shot.Hot => hot,
            Shot.Winner => winner,
            _ => wide,
        };

        void SetOffset(CinemachineCamera cam, Transform target, Vector3 offset)
        {
            if (cam == null) return;
            offset.y = Mathf.Min(offset.y, maxCameraHeight - target.position.y);
            var follow = cam.GetComponent<CinemachineFollow>();
            if (follow != null) follow.FollowOffset = offset;
        }

        // ---------------------------------------------------------------- original direct framing (no rig)
        void Legacy(IBroadcastBoard b, List<((int place, string name, string result, bool bad, PolicyRunner runner) r, Transform t)> bodies,
                    Vector3 focus, Vector3 f, Vector3 left)
        {
            Vector3 want, look = focus;
            switch (b.BoardState)
            {
                case BoardPhase.Result:
                    var w = bodies.FirstOrDefault(x => x.r.place == 1);
                    var wp = (w.t != null ? w.t : bodies[0].t).position;
                    look = new Vector3(wp.x, wp.y + 0.2f, wp.z);
                    // front three-quarter close-up: ahead of the winner, off to its right, just above head height
                    want = look + f * 5.5f - left * 3.2f + Vector3.up * 1.4f;
                    break;
                case BoardPhase.Live when kind == Kind.Race:
                    if (_shotClock > shotSeconds) { _shotClock = 0; _shot = (_shot + 1) % 3; _cut = true; }
                    want = _shot switch
                    {
                        1 => focus + f * 9f + Vector3.up * 1.6f - left * 2.0f,          // head-on, looking back at the pack
                        2 => focus - f * 11f + Vector3.up * 7f - left * 6f,            // high and wide from behind
                        _ => focus + establishing,                                    // trackside (establishing offset)
                    };
                    break;
                case BoardPhase.Live:
                    if (_shotClock > shotSeconds * 1.5f) _orbit += orbitSpeed * Time.unscaledDeltaTime;
                    want = focus + Quaternion.AngleAxis(Mathf.Rad2Deg * _orbit, Vector3.up) * establishing;
                    break;
                default:
                    _orbit = 0;
                    want = focus + establishing;
                    break;
            }
            want.y = Mathf.Min(want.y, maxCameraHeight);
            float k = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
            bool cut = !_init || _cut;             // hard cut between live race shots, smooth moves otherwise
            _cut = false;
            _pos = cut ? want : Vector3.Lerp(_pos, want, k);
            _look = cut ? look : Vector3.Lerp(_look, look, k);
            _init = true;
            transform.position = _pos;
            transform.rotation = Quaternion.LookRotation(_look - _pos, Vector3.up);
        }
    }
}
