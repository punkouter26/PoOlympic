using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 3 Deep Squat Endurance — 8 athletes on the venue's 2 x 4 station grid (scene_squat8.xml, no contact), Rung S brain
    /// (contract v4 pelvis-height command). Mirror of training/poolympic/events/squat.py: a metronome calls `reps` squat
    /// reps that get deeper (depthStart + depthStep·r, capped at depthMax) and faster (down / up phases 0.1 s shorter per
    /// rep). Rep points = repPoints × clamp(1 − |pelvis drop − depth| / depthTol, 0, 1), the drop averaged over the last
    /// `sampleFrac` of the down phase. Out = a fall (the fall line drops with the squat target − 0.10 m, like training)
    /// or a foot sliding more than `stepTol` from its start spot (STEPPED). Finishing every rep in balance earns
    /// `balanceBonus`. Rank by points; ties: the later exit first.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class DeepSquatEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Squatter
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float points, outAt = -1, lastErr = -1, drop;
            [NonSerialized] public int repsScored, place, footL = -1, footR = -1;
            [NonSerialized] public double z0, fallBase;
            [NonSerialized] public Vector2 startL, startR;
            [NonSerialized] public readonly List<float> buf = new();
            public bool In => status.Length == 0;
        }

        public List<Squatter> squatters = new();

        public void DropInactiveLanes() => squatters.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);

        [Header("Rules (= squat.py)")]
        public int reps = 12;
        public float depthStart = 0.25f, depthStep = 0.02f, depthMax = 0.45f;
        public float downStart = 2.5f, upStart = 2.0f, tempoStep = 0.1f, downMin = 1.2f, upMin = 1.0f;
        public float sampleFrac = 0.4f, repPoints = 10f, depthTol = 0.10f, stepTol = 0.20f, balanceBonus = 10f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public int Rep { get; private set; } = -1;          // current rep (down or up phase), -1 before GO
        public bool Down { get; private set; }
        public float Target { get; private set; }            // pelvis-height command now (≤ 0)
        public float MaxPoints => reps * repPoints + balanceBonus;

        public float Depth(int r) => Mathf.Min(depthMax, depthStart + depthStep * r);
        public float DownS(int r) => Mathf.Max(downMin, downStart - tempoStep * r);
        public float UpS(int r) => Mathf.Max(upMin, upStart - tempoStep * r);
        public float TotalS { get { float t = 0; for (int r = 0; r < reps; r++) t += DownS(r) + UpS(r); return t; } }

        public IEnumerable<Squatter> Standings => Current == Phase.Result
            ? squatters.OrderBy(s => s.place)
            : squatters.OrderByDescending(s => s.points).ThenBy(s => s.In ? 0 : 1).ThenByDescending(s => s.outAt);

        System.Random _rng;
        int _liveStartTick, _seg = -1;
        bool _traitsPending = true;

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var s in squatters)
            {
                s.status = ""; s.points = 0; s.outAt = s.lastErr = -1; s.repsScored = 0; s.place = 0; s.buf.Clear();
                s.runner.command = Vector3.zero;
                s.runner.laneKeeping = false;
                s.runner.skill = default;
                if (!first) s.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            Rep = -1; _seg = -1; Down = false; Target = 0;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = squatters[0].runner.Contract.trait_ranges;
            for (int i = 0; i < squatters.Count; i++)
            {
                var r = squatters[i].runner;
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

        static double DefaultRootZ(Contract c)
        {
            foreach (var j in c.default_joint_qpos)
                if (j.joint == c.root_joint) return j.qpos[2];
            throw new InvalidOperationException("contract: root joint missing");
        }

        /// <summary>Metronome segment at `live` s after GO: index 2r = rep r down, 2r + 1 = rep r up; -1 before / after.</summary>
        int Segment(float live, out float start, out float dur)
        {
            start = dur = 0;
            if (live < 0) return -1;
            float t = 0;
            for (int r = 0; r < reps; r++)
            {
                if (live < t + DownS(r) - 1e-4f) { start = t; dur = DownS(r); return 2 * r; }
                t += DownS(r);
                if (live < t + UpS(r) - 1e-4f) { start = t; dur = UpS(r); return 2 * r + 1; }
                t += UpS(r);
            }
            return -1;
        }

        unsafe void Update()
        {
            if (squatters.Count == 0 || squatters.Any(s => s.runner == null || !s.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var s in squatters)
            {
                if (s.judge != null) continue;
                s.judge = new AthleteJudge(m, s.runner, "ground");
                s.fallBase = s.runner.Contract.FallPelvisZ;
                s.z0 = DefaultRootZ(s.runner.Contract);
                var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
                s.footL = bodies[s.runner.athletePrefix + "foot_l"];
                s.footR = bodies[s.runner.athletePrefix + "foot_r"];
            }
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = squatters[0].runner;
            float tick = (float)(lead.Contract.timestep * lead.Contract.decimation);
            foreach (var s in squatters) s.drop = (float)(s.z0 - s.runner.PelvisHeight(d));
            switch (Current)
            {
                case Phase.Ready:
                    if (HoldStart) { PhaseTime = 0; break; }   // betting window (BroadcastHud)
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        foreach (var s in squatters)
                        {
                            s.startL = FootXY(d, s.footL);
                            s.startR = FootXY(d, s.footR);
                        }
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    int seg = Segment(LiveTime, out float start, out float dur);
                    if (seg != _seg)
                    {
                        if (_seg >= 0 && _seg % 2 == 0) ScoreRep(_seg / 2);
                        _seg = seg;
                        foreach (var s in squatters) s.buf.Clear();
                    }
                    if (seg < 0) { Finish(); break; }
                    Rep = seg / 2;
                    Down = seg % 2 == 0;
                    Target = Down ? -Depth(Rep) : 0f;
                    foreach (var s in squatters.Where(s => s.In))
                    {
                        s.runner.skill.pelvisHeight = Target;
                        s.judge.fallPelvisZ = s.fallBase + Mathf.Min(0f, Target) - (Target < 0f ? 0.10 : 0.0);
                        string why = s.judge.Eliminated(m, d);
                        if (why == null && (Vector2.Distance(FootXY(d, s.footL), s.startL) > stepTol ||
                                            Vector2.Distance(FootXY(d, s.footR), s.startR) > stepTol)) why = "STEPPED";
                        if (why != null) { Out(s, why == "STEPPED" ? why : "FELL"); continue; }
                        if (Down && LiveTime > start + dur * (1f - sampleFrac)) s.buf.Add(s.drop);
                    }
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        static unsafe Vector2 FootXY(MujocoLib.mjData_* d, int body) => new((float)d->xpos[3 * body], (float)d->xpos[3 * body + 1]);

        void ScoreRep(int rep)
        {
            float depth = Depth(rep);
            foreach (var s in squatters.Where(s => s.In && s.buf.Count > 0))
            {
                float err = Mathf.Abs(s.buf.Average() - depth);
                s.lastErr = err;
                s.points += repPoints * Mathf.Clamp01(1f - err / depthTol);
                s.repsScored++;
            }
        }

        void Out(Squatter s, string status)
        {
            s.status = status;
            s.outAt = LiveTime;
            s.runner.skill.pelvisHeight = 0f;
        }

        void Finish()
        {
            foreach (var s in squatters.Where(s => s.In)) { s.status = "DONE"; s.points += balanceBonus; }
            int p = 1;
            foreach (var s in squatters.OrderByDescending(s => s.points).ThenByDescending(s => s.outAt < 0 ? 1e9f : s.outAt)) s.place = p++;
            foreach (var s in squatters) s.runner.skill.pelvisHeight = 0f;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = squatters.First(s => s.place == 1);
            Debug.Log($"[DeepSquat] heat {Attempt} seed {seed + Attempt}: winner {w.name} {w.points:0.0} pts ({w.status}) after {LiveTime:F1} s");
        }

        public string Describe(Squatter s) => s.status switch
        {
            "" => $"{s.points:0.0} pts" + (s.lastErr >= 0 ? $"  {s.lastErr * 100f:0} cm" : ""),
            "DONE" => $"{s.points:0.0} pts",
            _ => $"{s.points:0.0} pts {s.status} {s.outAt:0.0} s",
        };

        // IBroadcastBoard (BroadcastHud: betting window, records)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = squatters.FirstOrDefault(s => s.place == 1);
            lowerIsBetter = false;
            value = 0; text = "";
            if (Current != Phase.Result || w == null) return false;
            value = w.points;
            text = $"{w.points:0.0} pts";
            return true;
        }

        public string SubtitleExtra => $"{reps} reps, {depthStart * 100f:0}-{depthMax * 100f:0} cm deep";
        public string ClockLine => Rep < 0 ? $"{LiveTime:0.0} s" : $"REP {Rep + 1}/{reps}  {(Down ? "DOWN" : "UP")}";
        public string InfoLine => Rep < 0 ? $"max {MaxPoints:0} pts" : $"target {Depth(Rep) * 100f:0} cm";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(s => (s.place, s.name, Describe(s), s.status is "FELL" or "STEPPED", s.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{squatters.First(x => x.place == 1).name} WINS\n{squatters.First(x => x.place == 1).points:0.0} pts",
            _ => LiveTime < 0.8f ? "SQUAT!" : "",
        };
    }
}
