using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Event 1 — The Iron Pedestal, official 8-runner heat (events_catalog.json): last runner standing wins.
    /// Mirror of training/poolympic/events/iron_pedestal.py: every round (3 s) each runner still in gets a gust of the same
    /// escalating magnitude (0.3 m/s + 0.05 per round) in its own seeded direction, and every 3rd round a 2 kg cube
    /// dropped from 1.5 m above the shoulder. Traits (strength / latency / sensor noise) are drawn per attempt from the
    /// contract ranges — the lanes' "form" that odds are built on. Ranking = elimination order.
    /// Crowd stage (scene_pedestal8 since 2026-09-30): all 8 stand shoulder to shoulder on ONE iron beam (geom `pedestal`)
    /// and collide — a gust into a neighbour can topple the row like dominoes.
    ///   Ready (countdown) → Live → Result → auto restart (new seed, new traits)
    /// </summary>
    public class IronPedestalHeat : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Runner
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public float outAt = -1;
            [NonSerialized] public string reason = "";
            [NonSerialized] public int gusts;
            [NonSerialized] public int place;
            public bool In => outAt < 0;
        }

        public List<Runner> runners = new();

        /// <summary>Roster scenes: drop the athletes LaneLineup switched off (list order stays lane order).</summary>
        public void DropInactiveLanes() => runners.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);
        public MjCubePool cubes;
        [Header("Rules (= iron_pedestal.py)")]
        public float countdownSeconds = 3f;
        public float roundSeconds = 3f;
        public float gustStart = 0.3f, gustStep = 0.05f;
        public int cubeEvery = 3;
        public float cubeDropHeight = 1.5f;
        public float maxSeconds = 120f;
        public float resultHoldSeconds = 6f;
        public bool autoRestart = true;
        public bool randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public int Round { get; private set; }
        public float GustNow => gustStart + gustStep * Mathf.Max(0, Round - 1);
        public int StillIn => runners.Count(r => r.In);
        public IEnumerable<Runner> Standings => runners.OrderBy(r => r.place == 0 ? 0 : r.place).ThenBy(r => r.In ? 0 : 1).ThenByDescending(r => r.outAt);

        System.Random _rng;
        int _liveStartTick, _nextRoundTick;

        float TickSeconds => (float)(runners[0].runner.Contract.timestep * runners[0].runner.Contract.decimation);
        int Ticks(float s) => Mathf.RoundToInt(s / TickSeconds);

        void Start() => BeginAttempt(first: true);

        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var r in runners)
            {
                r.outAt = -1; r.reason = ""; r.gusts = 0; r.place = 0;
                if (!first) r.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            Round = 0;
            _traitsPending = true;
        }

        bool _traitsPending;

        void DrawTraits()
        {
            var c = runners[0].runner.Contract;
            var tr = c.trait_ranges;
            for (int i = 0; i < runners.Count; i++)
            {
                var r = runners[i];
                if (!randomTraits || tr == null) { r.runner.SetTraits(1f, 0, 0f, 0); continue; }
                float str = (float)(tr.strength[0] + _rng.NextDouble() * (tr.strength[1] - tr.strength[0]));
                int lat = _rng.Next((int)tr.latency_substeps[0], (int)tr.latency_substeps[1] + 1);
                float noise = (float)(tr.obs_noise[0] + _rng.NextDouble() * (tr.obs_noise[1] - tr.obs_noise[0]));
                r.runner.SetTraits(str, lat, noise, seed * 1000 + Attempt * 16 + i);
            }
        }

        unsafe void Update()
        {
            if (runners.Count == 0 || runners.Any(r => r.runner == null || !r.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var r in runners)
                r.judge ??= new AthleteJudge(m, r.runner, "ground", r.runner.athletePrefix + "pedestal", "pedestal");   // own pedestal or the shared beam
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = runners[0].runner;
            switch (Current)
            {
                case Phase.Ready:
                    if (HoldStart) { PhaseTime = 0; break; }   // betting window (BroadcastHud)
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        _nextRoundTick = _liveStartTick + Ticks(1f);   // = COUNTDOWN_S settle before the first gust
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * TickSeconds;
                    if (lead.ControlTick >= _nextRoundTick)
                    {
                        Round++;
                        float dv = GustNow;
                        foreach (var r in runners.Where(r => r.In))
                        {
                            double a = _rng.NextDouble() * 2 * Math.PI;
                            cubes.Shove(r.runner, new Vector2((float)Math.Cos(a), (float)Math.Sin(a)) * dv);
                            r.gusts++;
                            if (Round % cubeEvery == 0) cubes.DropOnAthlete(r.runner, cubeDropHeight);
                        }
                        _nextRoundTick += Ticks(roundSeconds);
                    }
                    if (LiveTime >= 1f)
                        foreach (var r in runners.Where(r => r.In))
                        {
                            var why = r.judge.Eliminated(m, d);
                            if (why != null) { r.outAt = LiveTime; r.reason = why; }
                        }
                    if (StillIn <= 1 || LiveTime >= maxSeconds) Finish();
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        // IStandingsBoard / IBroadcastBoard (BroadcastHud)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public string SubtitleExtra => $"round {Round} · gust {GustNow:0.00} m/s";
        public string ClockLine => $"{LiveTime:0.0} s";
        public string InfoLine => $"{StillIn} of {runners.Count} still standing";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(x => (x.place, x.name,
                x.In ? (Current == Phase.Result ? "WINNER" : "IN") : $"{(x.reason == "STEPPED OFF" ? "OFF" : "FELL")} {x.outAt:0.0} s",
                !x.In, x.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{string.Join(" & ", runners.Where(x => x.place == 1).Select(x => x.name))} WINS\n{LiveTime:0.0} s",
            _ => LiveTime < 0.8f ? "GO!" : "",
        };
        /// <summary>Record = how long the last one stood (the heat's length).</summary>
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            lowerIsBetter = false;
            value = LiveTime; text = $"{LiveTime:0.0} s standing";
            return Current == Phase.Result;
        }

        void Finish()
        {
            var order = runners.OrderBy(r => r.In ? 0 : 1).ThenByDescending(r => r.outAt).ToList();
            int place = 1;
            for (int i = 0; i < order.Count; i++)
            {
                bool tie = i > 0 && ((order[i].In && order[i - 1].In) || (!order[i].In && Mathf.Approximately(order[i].outAt, order[i - 1].outAt)));
                if (!tie) place = i + 1;
                order[i].place = place;
            }
            Current = Phase.Result;
            PhaseTime = 0;
            var win = order.Where(r => r.place == 1).Select(r => r.name);
            Debug.Log($"[IronPedestalHeat] attempt {Attempt} seed {seed + Attempt}: winner {string.Join(", ", win)} after {LiveTime:F1} s, {Round} rounds");
        }
    }
}
