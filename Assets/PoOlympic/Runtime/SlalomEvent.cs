using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 11 Slalom Sprint — 8 runners weave through 7 physical poles on their lane's centre line (scene_slalom8.xml, Rung 2
    /// brain). Mirror of training/poolympic/events/slalom.py: poles at x = poleX0 + g·poleDx on y = 0; pole g passed on
    /// the left for even g. Racing line y*(x) = A·cos(π(x − poleX0)/poleDx) (0 outside half a gap around the poles); A =
    /// the runner's seeded "line" — wide lines avoid clipping poles but may topple the runner. Every control tick
    /// (PolicyRunner.steer): vx = speed, wz steers onto the line (heading + look-ahead), |wz| ≤ 4 m/s² / speed.
    /// Penalties: wrong side of a pole, pole contacts; a fall = out. Rank by time + penalties.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class SlalomEvent : MonoBehaviour, IStandingsBoard
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Racer
        {
            public PolicyRunner runner;
            public string name;
            public int lane;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float finishS = -1, score, x, y, line;
            [NonSerialized] public int misses, clips, nextPole, place;
            [NonSerialized] public bool touching;
            [NonSerialized] public int pelvis;
            [NonSerialized] public readonly HashSet<int> poles = new();
            public bool Racing => status.Length == 0;
        }

        public List<Racer> racers = new();
        [Header("Rules (= slalom.py)")]
        public float distance = 32f;
        public float poleX0 = 3f, poleDx = 4f;
        public int nPoles = 7;
        public float speed = 2.2f;
        public Vector2 lineRange = new(0.40f, 0.56f);
        public float lookahead = 0.8f, yGain = 1f, headingGain = 2f, maxLateralAccel = 4f;
        public float missPenalty = 2f, clipPenalty = 0.5f;
        public float maxSeconds = 30f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public IEnumerable<Racer> Standings => Current == Phase.Result ? racers.OrderBy(r => r.place) : racers.OrderByDescending(r => r.x);

        System.Random _rng;
        int _liveStartTick;
        bool _traitsPending = true, _bound;

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        /// <summary>= slalom.py line_y: racing line and its slope.</summary>
        public (double y, double slope) Line(double x, double a)
        {
            double lo = poleX0 - poleDx / 2, hi = poleX0 + (nPoles - 1) * poleDx + poleDx / 2;
            if (x < lo || x > hi) return (0, 0);
            double w = Math.PI / poleDx;
            return (a * Math.Cos(w * (x - poleX0)), -a * w * Math.Sin(w * (x - poleX0)));
        }

        /// <summary>= slalom.py slalom_command.</summary>
        public Vector3 Steer(double x, double y, double yaw, double a)
        {
            var (ys, _) = Line(x, a);
            var (_, slope) = Line(x + lookahead, a);
            double target = Math.Atan(slope) + Math.Atan(-yGain * (y - ys));
            double err = target - yaw + Math.PI;
            err = err - 2 * Math.PI * Math.Floor(err / (2 * Math.PI)) - Math.PI;
            double lim = maxLateralAccel / speed;
            return new Vector3(speed, 0f, (float)Math.Clamp(headingGain * err, -lim, lim));
        }

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var r in racers)
            {
                r.status = ""; r.finishS = -1; r.score = r.x = r.y = 0; r.misses = r.clips = r.nextPole = r.place = 0;
                r.touching = false;
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
                var r = racers[i];
                if (randomTraits && tr != null)
                {
                    float str = (float)(tr.strength[0] + _rng.NextDouble() * (tr.strength[1] - tr.strength[0]));
                    int lat = _rng.Next((int)tr.latency_substeps[0], (int)tr.latency_substeps[1] + 1);
                    float noise = (float)(tr.obs_noise[0] + _rng.NextDouble() * (tr.obs_noise[1] - tr.obs_noise[0]));
                    r.runner.SetTraits(str, lat, noise, seed * 1000 + Attempt * 16 + i);
                }
                else r.runner.SetTraits(1f, 0, 0f, 0);
                r.line = lineRange.x + (float)_rng.NextDouble() * (lineRange.y - lineRange.x);
            }
        }

        unsafe void Bind(MujocoLib.mjModel_* m)
        {
            var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
            var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);
            foreach (var r in racers)
            {
                r.pelvis = bodies[r.runner.athletePrefix + "pelvis"];
                r.poles.Clear();
                for (int g = 0; g < nPoles; g++) r.poles.Add(geoms[$"pole{r.lane}_{g}"]);
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
            foreach (var r in racers)
            {
                int ra = r.runner.Binding.RootQposAdr;
                r.x = (float)(d->qpos[ra] - r.runner.laneOriginX);
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
                        foreach (var r in racers)
                        {
                            double a = r.line;
                            r.runner.steer = (x, y, yaw, _) => Steer(x, y, yaw, a);
                        }
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    foreach (var r in racers.Where(r => r.Racing))
                    {
                        if (r.judge.Eliminated(m, d) == "FELL") { Out(r, "FELL"); continue; }
                        bool touching = false;
                        for (int i = 0; i < d->ncon && !touching; i++)
                        {
                            var c = d->contact[i];
                            int other = r.poles.Contains(c.geom1) ? c.geom2 : r.poles.Contains(c.geom2) ? c.geom1 : -1;
                            touching = other >= 0 && m->body_rootid[m->geom_bodyid[other]] == r.pelvis;
                        }
                        if (touching && !r.touching) r.clips++;
                        r.touching = touching;
                        while (r.nextPole < nPoles && r.x >= poleX0 + r.nextPole * poleDx)
                        {
                            if ((r.y > 0) != (r.nextPole % 2 == 0)) r.misses++;
                            r.nextPole++;
                        }
                        if (r.x >= distance) { r.status = "FINISHED"; r.finishS = LiveTime; Stand(r); }
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
            foreach (var r in racers.Where(r => r.status == "FINISHED")) r.score = r.finishS + missPenalty * r.misses + clipPenalty * r.clips;
            int p = 1;
            foreach (var r in racers.OrderBy(r => r.status == "FINISHED" ? 0 : 1).ThenBy(r => r.status == "FINISHED" ? r.score : 0f)) r.place = p++;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = racers.First(r => r.place == 1);
            Debug.Log($"[Slalom] heat {Attempt} seed {seed + Attempt}: winner {w.name} ({Describe(w)}) after {LiveTime:F1} s");
        }

        public string Describe(Racer r)
        {
            string faults = r.clips + r.misses > 0 ? $" +{r.clips}c" + (r.misses > 0 ? $" +{r.misses}m" : "") : "";
            string line = $"A{r.line * 100f:0}";
            return r.status switch
            {
                "FINISHED" when Current == Phase.Result => $"{r.score:0.00}" + (faults.Length > 0 ? $" ({r.finishS:0.00} s{faults})" : " s") + $" {line}",
                "FINISHED" => $"{r.finishS:0.00} s{faults} {line}",
                "" => $"{r.x:0.0} m{faults} {line}",
                _ => $"{r.status} {line}",
            };
        }

        public string SubtitleExtra => $"{nPoles} poles · {distance:0} m";
        public string ClockLine => $"{LiveTime:0.00} s";
        public string InfoLine => $"+{clipPenalty:0.#} s clip · +{missPenalty:0.#} s miss";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(r => (r.place, r.name, Describe(r), r.status is "FELL" or "DNF", r.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{racers.First(x => x.place == 1).name} WINS\n{racers.First(x => x.place == 1).score:0.00}",
            _ => LiveTime < 0.8f ? "GO!" : "",
        };
    }
}
