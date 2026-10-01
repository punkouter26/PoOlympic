using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 7 Cadence March — 8 athletes march in place on the venue's 2 x 4 station grid to a rising metronome, Rung S brain
    /// (contract v4 march command: the cadence drives the gait clock at zero velocity, knee lift in metres). Mirror of
    /// training/poolympic/events/march.py: `stages` stages of `stageSeconds`; stage s marches at hzStart + hzStep·s
    /// strides per second (one stride = one lift of each knee) with the knees called `lift` metres above standing.
    ///   a lift       = one excursion of a knee above half the called lift; it counts for the stage in which it ends
    ///   stage points = stagePoints × rhythm × height,
    ///                  rhythm = clamp(1 − |lifts − expected| / (rhythmTol × expected)), expected = 2 × cadence × stageSeconds
    ///                  height = clamp(mean peak knee rise / lift)
    ///   out          = a fall (FELL) or the pelvis drifting more than `driftOut` from its spot (WANDERED)
    /// Finishing every stage on the spot earns `finishBonus`. Rank by points; ties: the later exit first.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class CadenceMarchEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Marcher
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float points, outAt = -1, drift, lastPeak = -1;
            [NonSerialized] public int place, stagesScored, lastLifts, shinL = -1, shinR = -1;
            [NonSerialized] public double kneeL0, kneeR0, spotX, spotY;
            [NonSerialized] public readonly KneeCounter left = new(), right = new();
            public bool In => status.Length == 0;
        }

        /// <summary>One knee: excursions above half the called lift, with the peak rise of each (march.py KneeCounter).</summary>
        public sealed class KneeCounter
        {
            bool _up;
            float _peak;
            public readonly List<float> Done = new();     // peaks of the lifts that ended since the last Clear

            public void Sample(float rise, float lift)
            {
                if (rise > 0.5f * lift) { _peak = _up ? Mathf.Max(_peak, rise) : rise; _up = true; }
                else if (_up) { Done.Add(_peak); _up = false; _peak = 0f; }
            }

            public void Reset() { _up = false; _peak = 0f; Done.Clear(); }
        }

        public List<Marcher> marchers = new();

        public void DropInactiveLanes() => marchers.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);

        [Header("Rules (= march.py)")]
        public int stages = 9;
        public float stageSeconds = 4f;
        public float hzStart = 1f, hzStep = 0.125f;
        public float lift = 0.25f;
        public float rhythmTol = 0.25f, stagePoints = 10f, finishBonus = 10f, driftOut = 0.60f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public int Stage { get; private set; } = -1;         // current stage, -1 before GO
        public float MaxPoints => stages * stagePoints + finishBonus;

        public float Cadence(int s) => hzStart + hzStep * s;
        public float ExpectedLifts(int s) => 2f * Cadence(s) * stageSeconds;

        /// <summary>Points of one stage from its lift count and mean peak knee rise (march.py stage_points).</summary>
        public float StagePoints(int lifts, float meanPeak, int s)
        {
            float exp = ExpectedLifts(s);
            float rhythm = Mathf.Clamp01(1f - Mathf.Abs(lifts - exp) / (rhythmTol * exp));
            return stagePoints * rhythm * Mathf.Clamp01(meanPeak / lift);
        }

        public IEnumerable<Marcher> Standings => Current == Phase.Result
            ? marchers.OrderBy(s => s.place)
            : marchers.OrderByDescending(s => s.points).ThenBy(s => s.In ? 0 : 1).ThenByDescending(s => s.outAt);

        System.Random _rng;
        int _liveStartTick;
        bool _traitsPending = true;

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var s in marchers)
            {
                s.status = ""; s.points = 0; s.outAt = s.lastPeak = -1; s.drift = 0; s.place = s.stagesScored = s.lastLifts = 0;
                s.left.Reset(); s.right.Reset();
                s.runner.command = Vector3.zero;
                s.runner.laneKeeping = false;
                s.runner.skill = default;
                if (!first) s.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            Stage = -1;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = marchers[0].runner.Contract.trait_ranges;
            for (int i = 0; i < marchers.Count; i++)
            {
                var r = marchers[i].runner;
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

        unsafe void Update()
        {
            if (marchers.Count == 0 || marchers.Any(s => s.runner == null || !s.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var s in marchers)
            {
                if (s.judge != null) continue;
                s.judge = new AthleteJudge(m, s.runner, "ground");
                var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
                s.shinL = bodies[s.runner.athletePrefix + "shin_l"];
                s.shinR = bodies[s.runner.athletePrefix + "shin_r"];
            }
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = marchers[0].runner;
            float tick = (float)(lead.Contract.timestep * lead.Contract.decimation);
            switch (Current)
            {
                case Phase.Ready:
                    if (HoldStart) { PhaseTime = 0; break; }   // betting window (BroadcastHud)
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        foreach (var s in marchers)            // standing knee heights + the spot, marked at GO
                        {
                            s.kneeL0 = d->xpos[3 * s.shinL + 2];
                            s.kneeR0 = d->xpos[3 * s.shinR + 2];
                            int r = s.runner.Binding.RootQposAdr;
                            s.spotX = d->qpos[r];
                            s.spotY = d->qpos[r + 1];
                        }
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    int stage = Mathf.FloorToInt((LiveTime + 1e-4f) / stageSeconds);
                    if (stage != Stage)
                    {
                        if (Stage >= 0) ScoreStage(Stage);
                        Stage = stage;
                    }
                    if (stage >= stages) { Stage = stages - 1; Finish(); break; }
                    foreach (var s in marchers.Where(s => s.In))
                    {
                        s.runner.skill.marchHz = Cadence(stage);
                        s.runner.skill.kneeLift = lift;
                        int r = s.runner.Binding.RootQposAdr;
                        double dx = d->qpos[r] - s.spotX, dy = d->qpos[r + 1] - s.spotY;
                        s.drift = (float)Math.Sqrt(dx * dx + dy * dy);
                        string why = s.judge.Eliminated(m, d) != null ? "FELL" : s.drift > driftOut ? "WANDERED" : null;
                        if (why != null) { Out(s, why); continue; }
                        s.left.Sample((float)(d->xpos[3 * s.shinL + 2] - s.kneeL0), lift);
                        s.right.Sample((float)(d->xpos[3 * s.shinR + 2] - s.kneeR0), lift);
                    }
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void ScoreStage(int stage)
        {
            foreach (var s in marchers)
            {
                int n = s.left.Done.Count + s.right.Done.Count;
                float mean = n > 0 ? (s.left.Done.Sum() + s.right.Done.Sum()) / n : 0f;
                s.left.Done.Clear();
                s.right.Done.Clear();
                if (!s.In) continue;                        // out during this stage: no points for it
                s.lastLifts = n;
                s.lastPeak = mean;
                s.points += StagePoints(n, mean, stage);
                s.stagesScored++;
            }
        }

        void Out(Marcher s, string status)
        {
            s.status = status;
            s.outAt = LiveTime;
            s.runner.skill.marchHz = s.runner.skill.kneeLift = 0f;
        }

        void Finish()
        {
            foreach (var s in marchers.Where(s => s.In)) { s.status = "DONE"; s.points += finishBonus; }
            int p = 1;
            foreach (var s in marchers.OrderByDescending(s => s.points).ThenByDescending(s => s.outAt < 0 ? 1e9f : s.outAt)) s.place = p++;
            foreach (var s in marchers) s.runner.skill.marchHz = s.runner.skill.kneeLift = 0f;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = marchers.First(s => s.place == 1);
            Debug.Log($"[CadenceMarch] heat {Attempt} seed {seed + Attempt}: winner {w.name} {w.points:0.0} pts ({w.status}), " +
                      $"{marchers.Count(s => s.status == "DONE")} finished, after {LiveTime:F1} s");
        }

        public string Describe(Marcher s) => s.status switch
        {
            "" => $"{s.points:0.0} pts" + (s.lastPeak >= 0 ? $"  {s.lastPeak * 100f:0} cm" : ""),
            "DONE" => $"{s.points:0.0} pts",
            _ => $"{s.points:0.0} pts {s.status} {s.outAt:0.0} s",
        };

        // IBroadcastBoard (BroadcastHud: betting window, records)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = marchers.FirstOrDefault(s => s.place == 1);
            lowerIsBetter = false;
            value = 0; text = "";
            if (Current != Phase.Result || w == null) return false;
            value = w.points;
            text = $"{w.points:0.0} pts";
            return true;
        }

        public string SubtitleExtra => $"{stages} stages, {hzStart:0.0}-{Cadence(stages - 1):0.0} strides/s";
        // short lines: the clock sits between the FPS pill and the menu button
        public string ClockLine => Stage < 0 ? $"{LiveTime:0.0} s" : $"STAGE {Stage + 1}/{stages}";
        public string InfoLine => Stage < 0 ? $"max {MaxPoints:0} pts" : $"{Cadence(Stage) * 120f:0} steps/min";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(s => (s.place, s.name, Describe(s), s.status is "FELL" or "WANDERED", s.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{marchers.First(x => x.place == 1).name} WINS\n{marchers.First(x => x.place == 1).points:0.0} pts",
            _ => LiveTime < 0.8f ? "MARCH!" : "",
        };
    }
}
