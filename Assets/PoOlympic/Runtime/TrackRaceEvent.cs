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
    ///   AllFours  (8  30m All Fours)       crawl brains (events/all_fours.py): start face down, get onto all fours during
    ///                                      the countdown, crawl 30 m; falls never eliminate (tumbles are counted), standing
    ///                                      up for more than standDqSeconds = DQ; rank by finish time
    ///   Steeplechase (13 Steeplechase Jog) 50 m at 3.5 m/s with the flight brain; scored on ground time = finish time −
    ///                                      hang time (flights: both feet off the ground ≥ minFlight, counted per physics
    ///                                      substep like track.py FootGait); rank by ground time, then finish time
    /// Runners steer with the contract's lane keeping (PolicyRunner.laneKeeping). Traits + nerve are drawn per heat.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class TrackRaceEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Mode { Dash, Terminal, Brake, Inverted, AllFours, Steeplechase }
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
            [NonSerialized] public int place, tumbles, torsoId = -1;
            [NonSerialized] public float standT, lambda = 1f;
            [NonSerialized] public bool wasOnFours;
            [NonSerialized] public FootGait gait;
            [NonSerialized] public float scoreS = float.NaN;   // steeplechase: ground time
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
        [Header("Steeplechase (= track.py steeple)")]
        public float minFlight = 0.02f;
        [Header("All fours (= all_fours.py)")]
        public float standDqSeconds = 1f;
        public Vector2 crawlBand = new(0.25f, 0.8f);   // pelvis height band (m) × λ
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public float LeaderX => runners.Count == 0 ? 0 : runners.Max(r => r.x);
        public IEnumerable<Runner> Standings => Current == Phase.Result ? runners.OrderBy(r => r.place) : runners.OrderByDescending(r => r.x);

        System.Random _rng;
        int _liveStartTick, _prevTick;
        bool _traitsPending = true;

        public static (float distance, float speed, float maxS) Defaults(Mode m) => m switch
        {
            Mode.Dash => (30f, 3.8f, 25f),
            Mode.Terminal => (84.39f, 4.0f, 45f),
            Mode.Inverted => (20f, -1.5f, 30f),
            Mode.AllFours => (30f, 1.2f, 60f),
            Mode.Steeplechase => (50f, 3.5f, 30f),
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
                r.tumbles = 0; r.standT = 0; r.wasOnFours = false;
                r.gait?.Clear(); r.scoreS = float.NaN;
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
            foreach (var r in runners)
            {
                r.judge ??= new AthleteJudge(m, r.runner, "ground");
                if (mode == Mode.Steeplechase && r.gait == null) r.gait = new FootGait(m, r.runner.athletePrefix, minFlight);
                if (r.torsoId < 0)
                {
                    r.torsoId = MujocoLib.mj_name2id(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, r.runner.athletePrefix + "torso");
                    double k = r.runner.Contract.SpeedScale;
                    r.lambda = (float)(k * k);
                }
            }
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            if (mode == Mode.Steeplechase && !_stepHooked) { MjScene.Instance.postUpdateEvent += OnPostStep; _stepHooked = true; }
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
                    if (HoldStart) { PhaseTime = 0; break; }   // betting window (BroadcastHud)
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = _prevTick = lead.ControlTick;
                        foreach (var r in runners) { r.runner.command = new Vector3(commandSpeed, 0f, 0f); r.runner.laneKeeping = true; }
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    float elapsed = (lead.ControlTick - _prevTick) * tick;     // control time since the last frame
                    _prevTick = lead.ControlTick;
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
                        if (mode == Mode.AllFours)
                        {
                            int ra = r.runner.Binding.RootQposAdr;
                            float z = (float)d->qpos[ra + 2];
                            float tilt = Mathf.Rad2Deg * Mathf.Acos(Mathf.Clamp((float)d->xmat[9 * r.torsoId + 8], -1f, 1f));
                            bool on4 = z > crawlBand.x * r.lambda && z < crawlBand.y * r.lambda && tilt > 50f;
                            if (r.wasOnFours && z < crawlBand.x * r.lambda) r.tumbles++;
                            r.wasOnFours = on4;
                            r.standT = z > crawlBand.y * r.lambda && tilt < 40f ? r.standT + elapsed : 0f;
                            if (r.standT > standDqSeconds) { Out(r, "DQ"); continue; }
                            if (r.x >= distance) { r.status = "FINISHED"; r.finishS = LiveTime; Stand(r); }
                            continue;
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

        bool _stepHooked;

        /// <summary>Steeplechase: foot contacts after every mj_step (track.py FootGait.step), racing runners only.</summary>
        unsafe void OnPostStep(object sender, MjStepArgs e)
        {
            if (Current != Phase.Live) return;
            foreach (var r in runners)
                if (r.Racing && r.gait != null) r.gait.Step(e.data);
        }

        void OnDestroy()
        {
            if (_stepHooked && MjScene.InstanceExists) MjScene.Instance.postUpdateEvent -= OnPostStep;
        }

        // IStandingsBoard / IBroadcastBoard (BroadcastHud)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public string SubtitleExtra => mode switch
        {
            Mode.Terminal => $"{distance:0.##} m · peak 1 s speed",
            Mode.Brake => $"stop before the red line at {distance:0} m",
            Mode.Inverted => $"{distance:0} m backwards",
            Mode.AllFours => $"{distance:0} m on hands and feet",
            Mode.Steeplechase => $"{distance:0} m · ground time = finish − air",
            _ => $"{distance:0} m",
        };
        public string ClockLine => $"{LiveTime:0.00} s";
        public string InfoLine => $"leader {LeaderX:0.0} / {distance:0.#} m";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(x => (x.place, x.name,
                Current == Phase.Result || !x.Racing ? Describe(x)
                    : mode == Mode.Steeplechase ? $"{x.x:0.0} m · air {x.gait?.Hang ?? 0:0.0} s" : $"{x.x:0.0} m · {x.v:0.0} m/s",
                x.status is "DQ" or "FELL", x.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{runners.First(x => x.place == 1).name} WINS\n{Describe(runners.First(x => x.place == 1))}",
            _ => LiveTime < 0.8f ? "GO!" : "",
        };
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = runners.FirstOrDefault(r => r.place == 1);
            value = 0; text = ""; lowerIsBetter = mode != Mode.Terminal;
            if (Current != Phase.Result || w == null) return false;
            switch (mode)
            {
                case Mode.Terminal: if (w.status == "FELL") return false; value = w.peakMps; text = $"{w.peakMps:0.00} m/s"; return true;
                case Mode.Brake: if (w.status != "STOPPED") return false; value = w.gapM; text = $"{w.gapM * 100f:0} cm short"; return true;
                case Mode.Steeplechase: if (w.status != "FINISHED") return false; value = w.scoreS; text = $"{w.scoreS:0.00} s ground"; return true;
                default: if (w.status != "FINISHED") return false; value = w.finishS; text = $"{w.finishS:0.00} s"; return true;
            }
        }

        void Out(Runner r, string status) { r.status = status; Stand(r); }
        void Stand(Runner r) { r.runner.command = Vector3.zero; r.runner.laneKeeping = false; }

        void Finish()
        {
            foreach (var r in runners.Where(r => r.Racing)) { r.status = "DNF"; Stand(r); }
            if (mode == Mode.Steeplechase)
                foreach (var r in runners.Where(r => r.status == "FINISHED")) r.scoreS = r.finishS - (float)r.gait.Hang;
            IEnumerable<Runner> order = mode switch
            {
                Mode.Dash or Mode.Inverted => runners.OrderBy(r => r.status == "FINISHED" ? 0 : 1).ThenBy(r => r.finishS),
                Mode.AllFours => runners.OrderBy(r => r.status == "FINISHED" ? 0 : r.status == "DNF" ? 1 : 2).ThenBy(r => r.finishS),
                Mode.Steeplechase => runners.OrderBy(r => r.status == "FINISHED" ? 0 : 1).ThenBy(r => r.status == "FINISHED" ? r.scoreS : 0f).ThenBy(r => r.finishS),
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
            Mode.AllFours => (r.status == "FINISHED" ? $"{r.finishS:0.00} s" : r.status == "DQ" ? "DQ (stood up)" : r.status)
                             + (r.tumbles > 0 ? $" · {r.tumbles} tumble" + (r.tumbles > 1 ? "s" : "") : ""),
            Mode.Terminal => $"{r.peakMps:0.00} m/s" + (r.status == "FELL" ? " FELL" : ""),
            Mode.Steeplechase => r.status == "FINISHED"
                ? (float.IsNaN(r.scoreS) ? $"{r.finishS:0.00} s" : $"{r.scoreS:0.00} s ground") + $" · air {r.gait.Hang:0.0} s"
                : r.status == "" ? $"air {r.gait?.Hang ?? 0:0.0} s · {r.gait?.Flights ?? 0} flights" : r.status,
            _ => r.status == "STOPPED" ? $"{r.gapM * 100f:0} cm short" : r.status == "DQ" ? "DQ (crossed)" : r.status,
        };
    }

    /// <summary>Steeplechase flights of one athlete (= track.py FootGait): after every mj_step, is any foot/toe geom of
    /// the athlete touching the ground? Both feet off for ≥ minFlight = one flight (contact chatter ignored).</summary>
    public sealed unsafe class FootGait
    {
        readonly int _ground;
        readonly int[] _feet;
        readonly int _minSubsteps;
        readonly double _dt;
        int _air;
        public int Flights { get; private set; }
        public double Hang { get; private set; }
        public double Longest { get; private set; }

        public FootGait(MujocoLib.mjModel_* m, string prefix, float minFlight)
        {
            var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);   // as AthleteJudge
            _ground = geoms["ground"];
            _feet = new[] { "foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0" }.Select(n => geoms[prefix + n]).ToArray();
            _dt = m->opt.timestep;
            _minSubsteps = (int)Math.Round(minFlight / _dt);
        }

        public void Clear() { _air = 0; Flights = 0; Hang = Longest = 0; }

        public void Step(MujocoLib.mjData_* d)
        {
            bool on = false;
            for (int i = 0; i < d->ncon && !on; i++)
            {
                var c = d->contact[i];
                on = (c.geom1 == _ground && Array.IndexOf(_feet, c.geom2) >= 0) || (c.geom2 == _ground && Array.IndexOf(_feet, c.geom1) >= 0);
            }
            if (!on) { _air++; return; }
            if (_air >= _minSubsteps)
            {
                Flights++;
                Hang += _air * _dt;
                Longest = Math.Max(Longest, _air * _dt);
            }
            _air = 0;
        }
    }
}
