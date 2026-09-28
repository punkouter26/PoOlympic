using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 12 The 360 Turntable — 8 athletes on the venue's spin spots (Rung 2 brain). Mirror of
    /// training/poolympic/events/turntable.py: on GO every athlete gets a pure in-place yaw command (`spinRate`, the top of
    /// the trained envelope) in the heat's direction until its pelvis heading has turned `turns` full circles, then zero
    /// command. Score = time + `driftPenalty` × largest pelvis drift from the spot; a fall = FELL, leaving the painted ring
    /// (`ringRadius`) = DQ. Traits + direction are drawn per heat.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class TurntableEvent : MonoBehaviour, IStandingsBoard
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Spinner
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float timeS = -1, maxDrift, drift, turned, score, prevYaw, rate;
            [NonSerialized] public int place;
            public bool Spinning => status.Length == 0;
        }

        public List<Spinner> spinners = new();
        [Header("Rules (= turntable.py)")]
        public float spinRate = 3.0f;
        public int turns = 3;
        public float driftPenalty = 2f, ringRadius = 1.1f;
        public float maxSeconds = 15f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public int Direction { get; private set; } = 1;
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public float Goal => turns * 2f * Mathf.PI;
        public IEnumerable<Spinner> Standings => Current == Phase.Result
            ? spinners.OrderBy(s => s.place)
            : spinners.OrderBy(s => s.status == "DONE" ? 0 : s.Spinning ? 1 : 2).ThenBy(s => s.status == "DONE" ? s.score : -s.turned);

        System.Random _rng;
        int _liveStartTick;
        bool _traitsPending = true;

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            Direction = _rng.NextDouble() < 0.5 ? 1 : -1;
            foreach (var s in spinners)
            {
                s.status = ""; s.timeS = -1; s.maxDrift = s.drift = s.turned = s.score = s.rate = 0; s.place = 0;
                s.runner.command = Vector3.zero;
                s.runner.laneKeeping = false;
                if (!first) s.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = spinners[0].runner.Contract.trait_ranges;
            for (int i = 0; i < spinners.Count; i++)
            {
                var r = spinners[i].runner;
                if (randomTraits && tr != null)
                {
                    float str = (float)(tr.strength[0] + _rng.NextDouble() * (tr.strength[1] - tr.strength[0]));
                    int lat = _rng.Next((int)tr.latency_substeps[0], (int)tr.latency_substeps[1] + 1);
                    float noise = (float)(tr.obs_noise[0] + _rng.NextDouble() * (tr.obs_noise[1] - tr.obs_noise[0]));
                    r.SetTraits(str, lat, noise, seed * 1000 + Attempt * 16 + i);
                }
                else r.SetTraits(1f, 0, 0f, 0);
            }
        }

        static float Yaw(double w, double x, double y, double z) => (float)Math.Atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

        unsafe void Update()
        {
            if (spinners.Count == 0 || spinners.Any(s => s.runner == null || !s.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var s in spinners) s.judge ??= new AthleteJudge(m, s.runner, "ground");
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = spinners[0].runner;
            float tick = (float)(lead.Contract.timestep * lead.Contract.decimation);
            foreach (var s in spinners)
            {
                int ra = s.runner.Binding.RootQposAdr;
                s.drift = (float)Math.Sqrt(Math.Pow(d->qpos[ra] - s.runner.laneOriginX, 2) + Math.Pow(d->qpos[ra + 1] - s.runner.laneOriginY, 2));
                float yaw = Yaw(d->qpos[ra + 3], d->qpos[ra + 4], d->qpos[ra + 5], d->qpos[ra + 6]);
                float step = Mathf.DeltaAngle(s.prevYaw * Mathf.Rad2Deg, yaw * Mathf.Rad2Deg) * Mathf.Deg2Rad;
                s.prevYaw = yaw;
                if (Current == Phase.Live && s.Spinning)
                {
                    s.turned += Direction * step;
                    s.rate = Direction * step / Mathf.Max(Time.deltaTime, 1e-4f);
                }
            }
            switch (Current)
            {
                case Phase.Ready:
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        foreach (var s in spinners) s.runner.command = new Vector3(0f, 0f, Direction * spinRate);
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    foreach (var s in spinners.Where(s => s.status is "" or "DONE"))
                    {
                        s.maxDrift = Mathf.Max(s.maxDrift, s.drift);
                        if (s.judge.Eliminated(m, d) == "FELL") { Out(s, "FELL"); continue; }
                        if (s.drift > ringRadius) { Out(s, "DQ"); continue; }
                        if (s.Spinning && s.turned >= Goal) { s.status = "DONE"; s.timeS = LiveTime; s.runner.command = Vector3.zero; }
                    }
                    if (spinners.All(s => !s.Spinning) || LiveTime >= maxSeconds) Finish();
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void Out(Spinner s, string status) { s.status = status; s.runner.command = Vector3.zero; }

        void Finish()
        {
            foreach (var s in spinners.Where(s => s.Spinning)) Out(s, "DNF");
            foreach (var s in spinners.Where(s => s.status == "DONE")) s.score = s.timeS + driftPenalty * s.maxDrift;
            int p = 1;
            foreach (var s in spinners.OrderBy(s => s.status == "DONE" ? 0 : 1).ThenBy(s => s.status == "DONE" ? s.score : 0f)) s.place = p++;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = spinners.First(s => s.place == 1);
            Debug.Log($"[Turntable] heat {Attempt} seed {seed + Attempt} dir {Direction}: winner {w.name} ({Describe(w)}) after {LiveTime:F1} s");
        }

        public string Describe(Spinner s) => s.status switch
        {
            "DONE" when Current == Phase.Result => $"{s.score:0.00} ({s.timeS:0.00} s, {s.maxDrift * 100f:0} cm)",
            "DONE" => $"{s.timeS:0.00} s  {s.maxDrift * 100f:0} cm",
            "DQ" => "DQ (left ring)",
            "" => $"{s.turned / (2f * Mathf.PI):0.00} turns",
            _ => s.status,
        };

        public string SubtitleExtra => $"{turns} turns {(Direction > 0 ? "anticlockwise" : "clockwise")}";
        public string ClockLine => $"{LiveTime:0.00} s";
        public string InfoLine => $"+{driftPenalty:0.#} s per m drift";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(s => (s.place, s.name, Describe(s), s.status is "DQ" or "FELL", s.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{spinners.First(x => x.place == 1).name} WINS\n{spinners.First(x => x.place == 1).score:0.00}",
            _ => LiveTime < 0.8f ? "SPIN!" : "",
        };
    }
}
