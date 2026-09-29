using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 10 Crab Shuffle — 8 athletes side-step `distance` between steel rails (scene_crab8.xml, Rung 2 brain). Mirror of
    /// training/poolympic/events/crab.py: the athletes face the course's left, so the course runs along their right
    /// (−y in the athlete frame). Every control tick (PolicyRunner.steer, = crab.py crab_command): vy = −sideSpeed,
    /// heading held square to the course, drift towards a rail corrected with vx. Penalties: every leg crossing (left
    /// foot to the right of the right foot in the pelvis frame) and every new rail contact add seconds; a fall = out.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class CrabShuffleEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Racer
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float finishS = -1, score, progress, maxDrift;
            [NonSerialized] public int crossings, railTouches, place;
            [NonSerialized] public bool crossed, touching;
            [NonSerialized] public int pelvis, footL, footR;
            public bool Racing => status.Length == 0;
        }

        public List<Racer> racers = new();

        /// <summary>Roster scenes: drop the athletes LaneLineup switched off (list order stays lane order).</summary>
        public void DropInactiveLanes() => racers.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);
        [Header("Rules (= crab.py)")]
        public float distance = 20f;
        public float sideSpeed = 1.2f;
        public float headingGain = 2f, wzLimit = 0.5f, xGain = 1f, vxLimit = 0.3f;
        public float crossPenalty = 1f, railPenalty = 1f;
        public float maxSeconds = 40f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;
        [Tooltip("MuJoCo geom names of the rails (crab8_layout.json props).")]
        public string[] railGeoms = Array.Empty<string>();

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public IEnumerable<Racer> Standings => Current == Phase.Result ? racers.OrderBy(r => r.place) : racers.OrderByDescending(r => r.progress);

        System.Random _rng;
        int _liveStartTick;
        bool _traitsPending = true, _bound;
        readonly HashSet<int> _rails = new();

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        /// <summary>= crab.py crab_command (x offset from the lane origin, heading) → (vx, vy, wz).</summary>
        public Vector3 Steer(double x, double y, double yaw, Vector3 command)
        {
            double err = -yaw + Math.PI;
            err = err - 2 * Math.PI * Math.Floor(err / (2 * Math.PI)) - Math.PI;
            double wz = Math.Clamp(headingGain * err, -wzLimit, wzLimit);
            double vx = Math.Clamp(-xGain * x, -vxLimit, vxLimit);
            return new Vector3((float)vx, -sideSpeed, (float)wz);
        }

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var r in racers)
            {
                r.status = ""; r.finishS = -1; r.score = r.progress = r.maxDrift = 0; r.crossings = r.railTouches = r.place = 0;
                r.crossed = r.touching = false;
                r.runner.command = Vector3.zero;
                r.runner.steer = null;
                if (!first) r.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = racers[0].runner.Contract.trait_ranges;
            for (int i = 0; i < racers.Count; i++)
            {
                var r = racers[i].runner;
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

        unsafe void Bind(MujocoLib.mjModel_* m)
        {
            var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
            var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);
            foreach (var n in railGeoms) _rails.Add(geoms[n]);
            foreach (var r in racers)
            {
                var p = r.runner.athletePrefix;
                r.pelvis = bodies[p + "pelvis"];
                r.footL = geoms[p + "foot_l_geom0"];
                r.footR = geoms[p + "foot_r_geom0"];
                r.judge = new AthleteJudge(m, r.runner, "ground");
            }
            _bound = true;
        }

        unsafe void Update()
        {
            if (racers.Count == 0 || racers.Any(r => r.runner == null || !r.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            if (!_bound) Bind(m);
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = racers[0].runner;
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
                        foreach (var r in racers) r.runner.steer = Steer;
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    var onRail = new HashSet<int>();
                    for (int i = 0; i < d->ncon; i++)
                    {
                        var c = d->contact[i];
                        int other = _rails.Contains(c.geom1) ? c.geom2 : _rails.Contains(c.geom2) ? c.geom1 : -1;
                        if (other >= 0) onRail.Add(m->body_rootid[m->geom_bodyid[other]]);
                    }
                    foreach (var r in racers.Where(r => r.Racing))
                    {
                        int ra = r.runner.Binding.RootQposAdr;
                        r.progress = -(float)(d->qpos[ra + 1] - r.runner.laneOriginY);
                        r.maxDrift = Mathf.Max(r.maxDrift, Mathf.Abs((float)(d->qpos[ra] - r.runner.laneOriginX)));
                        if (r.judge.Eliminated(m, d) == "FELL") { Out(r, "FELL"); continue; }
                        // left foot to the right of the right foot, along the pelvis' left axis (xmat column 1)
                        double dl = 0;
                        for (int k = 0; k < 3; k++)
                            dl += (d->geom_xpos[3 * r.footL + k] - d->geom_xpos[3 * r.footR + k]) * d->xmat[9 * r.pelvis + 3 * k + 1];
                        bool crossed = dl < 0;
                        if (crossed && !r.crossed) r.crossings++;
                        r.crossed = crossed;
                        bool touching = onRail.Contains(r.pelvis);
                        if (touching && !r.touching) r.railTouches++;
                        r.touching = touching;
                        if (r.progress >= distance) { r.status = "FINISHED"; r.finishS = LiveTime; Stand(r); }
                    }
                    if (racers.All(r => !r.Racing) || LiveTime >= maxSeconds) Finish();
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void Out(Racer r, string status) { r.status = status; Stand(r); }
        void Stand(Racer r) { r.runner.steer = null; r.runner.command = Vector3.zero; }

        void Finish()
        {
            foreach (var r in racers.Where(r => r.Racing)) Out(r, "DNF");
            foreach (var r in racers.Where(r => r.status == "FINISHED")) r.score = r.finishS + crossPenalty * r.crossings + railPenalty * r.railTouches;
            int p = 1;
            foreach (var r in racers.OrderBy(r => r.status == "FINISHED" ? 0 : 1).ThenBy(r => r.status == "FINISHED" ? r.score : 0f)) r.place = p++;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = racers.First(r => r.place == 1);
            Debug.Log($"[CrabShuffle] heat {Attempt} seed {seed + Attempt}: winner {w.name} ({Describe(w)}) after {LiveTime:F1} s");
        }

        public string Describe(Racer r)
        {
            string faults = r.crossings + r.railTouches > 0 ? $" +{r.crossings}x +{r.railTouches}r" : "";
            return r.status switch
            {
                "FINISHED" when Current == Phase.Result => $"{r.score:0.00}" + (faults.Length > 0 ? $" ({r.finishS:0.00} s{faults})" : " s"),
                "FINISHED" => $"{r.finishS:0.00} s{faults}",
                "" => $"{r.progress:0.0} m{faults}",
                _ => r.status,
            };
        }

        // IBroadcastBoard (BroadcastHud: betting window, records)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = racers.FirstOrDefault(r => r.place == 1);
            lowerIsBetter = true;
            value = 0; text = "";
            if (Current != Phase.Result || w == null || !(w.status == "FINISHED")) return false;
            value = w.score;
            text = $"{w.score:0.00} s";
            return true;
        }

        public string SubtitleExtra => $"{distance:0} m side-step between rails";
        public string ClockLine => $"{LiveTime:0.00} s";
        public string InfoLine => $"+{crossPenalty:0.#} s cross · +{railPenalty:0.#} s rail";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(r => (r.place, r.name, Describe(r), r.status is "FELL" or "DNF", r.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{racers.First(x => x.place == 1).name} WINS\n{Describe(racers.First(x => x.place == 1))}",
            _ => LiveTime < 0.8f ? "GO!" : "",
        };
    }
}
