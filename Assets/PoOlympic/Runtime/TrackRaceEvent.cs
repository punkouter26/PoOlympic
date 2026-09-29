using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Straight-track races for 8 runners (Rung 2 brain) — mirror of training/poolympic/events/track.py:
    ///   Dash      (8  The 30m Dash)        sprint `distance` from standstill; rank by finish time
    ///   Terminal  (19 Terminal Velocity)   open sprint of the back straight at the max trained command; rank by peak 1 s speed
    ///   Brake     (22 Emergency Brake)     run in at 3 m/s; each runner brakes (command 0) at its seeded "nerve" distance
    ///                                      before the red line; toe past the line = DQ; rank by the gap left
    ///   Inverted  (9  The Inverted Sprint)  20 m backwards (command vx &lt; 0; runners face away from the finish, the scene
    ///                                      turns the stadium 180°); pelvis more than `laneHalf` off the lane line = DQ
    /// Runners steer with the contract's lane keeping (PolicyRunner.laneKeeping). Traits + nerve are drawn per heat.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class TrackRaceEvent : MonoBehaviour, ILaneRoster
    {
        public enum Mode { Dash, Terminal, Brake, Inverted }
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Runner
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float finishS = -1, peakMps, gapM = float.NaN, nerveM, x, v, y;
            [NonSerialized] public bool braking;
            [NonSerialized] public int place;
            [NonSerialized] public readonly Queue<float> speedWindow = new();
            public bool Racing => status.Length == 0;
        }

        public Mode mode = Mode.Dash;
        public List<Runner> runners = new();

        /// <summary>Roster scenes: drop the athletes LaneLineup switched off (list order stays lane order).</summary>
        public void DropInactiveLanes() => runners.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);
        [Header("Rules (= track.py MODES)")]
        public float distance = 30f;
        public float commandSpeed = 3.8f;
        public float maxSeconds = 25f;
        public Vector2 brakeNerve = new(1.4f, 2.2f);
        public float toeAhead = 0.25f, stoppedSpeed = 0.1f, laneHalf = 0.61f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public float LeaderX => runners.Count == 0 ? 0 : runners.Max(r => r.x);
        public IEnumerable<Runner> Standings => Current == Phase.Result ? runners.OrderBy(r => r.place) : runners.OrderByDescending(r => r.x);

        System.Random _rng;
        int _liveStartTick;
        bool _traitsPending = true;

        public static (float distance, float speed, float maxS) Defaults(Mode m) => m switch
        {
            Mode.Dash => (30f, 3.8f, 25f),
            Mode.Terminal => (84.39f, 4.0f, 45f),
            Mode.Inverted => (20f, -1.5f, 30f),
            _ => (30f, 3.0f, 25f),
        };

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var r in runners)
            {
                r.status = ""; r.finishS = -1; r.peakMps = 0; r.gapM = float.NaN; r.braking = false; r.place = 0;
                r.speedWindow.Clear();
                r.runner.command = Vector3.zero;
                r.runner.laneKeeping = false;
                if (!first) r.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = runners[0].runner.Contract.trait_ranges;
            for (int i = 0; i < runners.Count; i++)
            {
                var r = runners[i];
                if (randomTraits && tr != null)
                {
                    float str = (float)(tr.strength[0] + _rng.NextDouble() * (tr.strength[1] - tr.strength[0]));
                    int lat = _rng.Next((int)tr.latency_substeps[0], (int)tr.latency_substeps[1] + 1);
                    float noise = (float)(tr.obs_noise[0] + _rng.NextDouble() * (tr.obs_noise[1] - tr.obs_noise[0]));
                    r.runner.SetTraits(str, lat, noise, seed * 1000 + Attempt * 16 + i);
                }
                else r.runner.SetTraits(1f, 0, 0f, 0);
                r.nerveM = brakeNerve.x + (float)_rng.NextDouble() * (brakeNerve.y - brakeNerve.x);
            }
        }

        unsafe void Update()
        {
            if (runners.Count == 0 || runners.Any(r => r.runner == null || !r.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var r in runners) r.judge ??= new AthleteJudge(m, r.runner, "ground");
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = runners[0].runner;
            float tick = (float)(lead.Contract.timestep * lead.Contract.decimation);
            float sign = commandSpeed < 0f ? -1f : 1f;   // progress is measured along the race direction
            foreach (var r in runners)
            {
                int ra = r.runner.Binding.RootQposAdr, da = r.runner.Binding.RootDofAdr;
                r.x = sign * (float)(d->qpos[ra] - r.runner.laneOriginX);
                r.v = sign * (float)d->qvel[da];
                r.y = (float)(d->qpos[ra + 1] - r.runner.laneOriginY);
            }
            switch (Current)
            {
                case Phase.Ready:
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        foreach (var r in runners) { r.runner.command = new Vector3(commandSpeed, 0f, 0f); r.runner.laneKeeping = true; }
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    foreach (var r in runners.Where(r => r.Racing))
                    {
                        r.speedWindow.Enqueue(r.v);
                        while (r.speedWindow.Count > Mathf.RoundToInt(1f / Mathf.Max(Time.deltaTime, 1e-3f))) r.speedWindow.Dequeue();
                        r.peakMps = Mathf.Max(r.peakMps, r.speedWindow.Average());
                        if (mode == Mode.Brake && !r.braking && r.x >= distance - r.nerveM)
                        {
                            r.braking = true;
                            r.runner.command = Vector3.zero;
                            r.runner.laneKeeping = false;
                        }
                        var why = r.judge.Eliminated(m, d);
                        if (why == "FELL") { Out(r, "FELL"); continue; }
                        if (mode == Mode.Inverted && Mathf.Abs(r.y) > laneHalf) { Out(r, "DQ"); continue; }
                        if (mode != Mode.Brake && r.x >= distance) { r.status = "FINISHED"; r.finishS = LiveTime; Stand(r); }
                        if (mode == Mode.Brake)
                        {
                            float toe = r.x + toeAhead;
                            if (toe > distance) { r.status = "DQ"; r.gapM = distance - toe; }
                            else if (r.braking && Mathf.Abs(r.v) < stoppedSpeed && LiveTime > 1f) { r.status = "STOPPED"; r.gapM = distance - toe; }
                        }
                    }
                    if (runners.All(r => !r.Racing) || LiveTime >= maxSeconds) Finish();
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void Out(Runner r, string status) { r.status = status; Stand(r); }
        void Stand(Runner r) { r.runner.command = Vector3.zero; r.runner.laneKeeping = false; }

        void Finish()
        {
            foreach (var r in runners.Where(r => r.Racing)) { r.status = "DNF"; Stand(r); }
            IEnumerable<Runner> order = mode switch
            {
                Mode.Dash or Mode.Inverted => runners.OrderBy(r => r.status == "FINISHED" ? 0 : 1).ThenBy(r => r.finishS),
                Mode.Terminal => runners.OrderBy(r => r.status == "FELL" ? 1 : 0).ThenByDescending(r => r.peakMps),
                _ => runners.OrderBy(r => r.status == "STOPPED" ? 0 : 1).ThenBy(r => r.status == "STOPPED" ? r.gapM : 0f),
            };
            int p = 1;
            foreach (var r in order) r.place = p++;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = runners.First(r => r.place == 1);
            Debug.Log($"[TrackRace] {mode} heat {Attempt} seed {seed + Attempt}: winner {w.name} ({Describe(w)}) after {LiveTime:F1} s");
        }

        public string Describe(Runner r) => mode switch
        {
            Mode.Dash => r.status == "FINISHED" ? $"{r.finishS:0.00} s" : r.status,
            Mode.Inverted => r.status == "FINISHED" ? $"{r.finishS:0.00} s" : r.status == "DQ" ? "DQ (left lane)" : r.status,
            Mode.Terminal => $"{r.peakMps:0.00} m/s" + (r.status == "FELL" ? " FELL" : ""),
            _ => r.status == "STOPPED" ? $"{r.gapM * 100f:0} cm short" : r.status == "DQ" ? "DQ (crossed)" : r.status,
        };
    }
}
