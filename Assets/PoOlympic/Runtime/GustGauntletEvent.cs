using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 5 The Gust Gauntlet — 8 athletes on spring-mounted shaker platforms (scene_shaker8.xml, Rung 2 brain). Mirror of
    /// training/poolympic/events/gauntlet.py: every control tick (PolicyRunner.steer) the athlete homes on its spot (walk
    /// command back, heading held). Rounds every `roundSeconds`: a lateral wind burst for everyone (root velocity kick,
    /// same size, own seeded side) and on every `shakeEvery`-th round a floor shake (platform velocity kick). Recovery per
    /// round = time until back within `calmRadius` and slower than `calmSpeed` for `calmSeconds` (cap = round length);
    /// a fall or a foot off the platform = out. Rank: still in by total recovery time, then eliminated (later = better).
    /// Crowd stage (scene_shaker8 since 2026-09-30): all 8 share ONE shaker floor (joints shaker_x/_y) in a tight 2 x 4
    /// grid — bursts push the rows into each other and a jolt throws everyone at once (one direction per round, drawn from
    /// its own per-heat stream); scenes with a platform per lane (L&lt;k&gt;_shaker) keep per-lane jolts.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class GustGauntletEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Athlete
        {
            public PolicyRunner runner;
            public string name;
            [Tooltip("Lane index: the shaker platform belongs to the lane (L<k>_shaker), whichever body stands on it.")]
            public int lane;
            public string ShakerPrefix => $"L{lane}_";
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public readonly List<float> recoveries = new();
            [NonSerialized] public float total, outAt = -1, burstAt = -1, calm, offset, speed;
            [NonSerialized] public string reason = "";
            [NonSerialized] public int place;
            public bool In => outAt < 0;
        }

        public List<Athlete> athletes = new();

        /// <summary>Roster scenes: drop the athletes LaneLineup switched off (list order stays lane order).</summary>
        public void DropInactiveLanes() => athletes.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);
        [Header("Rules (= gauntlet.py)")]
        public float startSeconds = 1f, roundSeconds = 4f;
        public int rounds = 10;
        public float gustStart = 0.5f, gustStep = 0.05f, gustSpreadDeg = 30f;
        public int shakeEvery = 2;
        public float shakeSpeed = 2f;
        public float homeGain = 1.5f, homeSpeed = 0.6f, homeDeadband = 0.08f;
        public float calmRadius = 0.15f, calmSpeed = 0.2f, calmSeconds = 0.5f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public int Round { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public float GustNow => gustStart + gustStep * Mathf.Max(0, Round - 1);
        public IEnumerable<Athlete> Standings => athletes.OrderBy(a => a.In ? 0 : 1).ThenBy(a => a.In ? a.total : -a.outAt);

        System.Random _rng, _floorRng;
        bool? _shared;                          // one shaker floor for everyone (joints shaker_x/_y)
        int _liveStartTick, _nextRoundTick;
        bool _traitsPending = true;

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        /// <summary>= gauntlet.py homing_command (offset from the spot, heading) → walk command back.</summary>
        public Vector3 Steer(double x, double y, double yaw, Vector3 command)
        {
            double c = Math.Cos(yaw), s = Math.Sin(yaw);
            double bx = c * x + s * y, by = -s * x + c * y;
            double vx = 0, vy = 0;
            if (Math.Sqrt(bx * bx + by * by) > homeDeadband)
            {
                vx = Math.Clamp(-homeGain * bx, -homeSpeed, homeSpeed);
                vy = Math.Clamp(-homeGain * by, -homeSpeed, homeSpeed);
            }
            return new Vector3((float)vx, (float)vy, (float)Math.Clamp(-2.0 * yaw, -0.5, 0.5));
        }

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            _floorRng = new System.Random((seed + Attempt) * 7919 + 5999);
            foreach (var a in athletes)
            {
                a.recoveries.Clear(); a.total = 0; a.outAt = a.burstAt = -1; a.calm = 0; a.reason = ""; a.place = 0;
                a.runner.command = Vector3.zero;
                a.runner.steer = null;
                if (!first) a.runner.RequestReset();
            }
            Current = Phase.Ready;
            Round = 0;
            PhaseTime = LiveTime = 0;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = athletes[0].runner.Contract.trait_ranges;
            for (int i = 0; i < athletes.Count; i++)
            {
                var r = athletes[i].runner;
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

        int Ticks(float seconds) => Mathf.RoundToInt(seconds / (float)(athletes[0].runner.Contract.timestep * athletes[0].runner.Contract.decimation));

        unsafe void Update()
        {
            if (athletes.Count == 0 || athletes.Any(a => a.runner == null || !a.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var a in athletes) a.judge ??= new AthleteJudge(m, a.runner, "ground", a.ShakerPrefix + "shaker", "shaker");
            if (_shared == null)
                _shared = ModelFingerprint.Id(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, "shaker_x") >= 0;
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = athletes[0].runner;
            float tick = (float)(lead.Contract.timestep * lead.Contract.decimation);
            foreach (var a in athletes)
            {
                int ra = a.runner.Binding.RootQposAdr, da = a.runner.Binding.RootDofAdr;
                double dx = d->qpos[ra] - a.runner.laneOriginX, dy = d->qpos[ra + 1] - a.runner.laneOriginY;
                a.offset = (float)Math.Sqrt(dx * dx + dy * dy);
                a.speed = (float)Math.Sqrt(d->qvel[da] * d->qvel[da] + d->qvel[da + 1] * d->qvel[da + 1]);
            }
            switch (Current)
            {
                case Phase.Ready:
                    if (HoldStart) { PhaseTime = 0; break; }   // betting window (BroadcastHud)
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        _nextRoundTick = _liveStartTick + Ticks(startSeconds);
                        foreach (var a in athletes) a.runner.steer = Steer;
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    if (Round < rounds && lead.ControlTick >= _nextRoundTick)
                    {
                        Round++;
                        float dv = GustNow;
                        bool shake = Round % shakeEvery == 0;
                        double fa = _floorRng.NextDouble() * 2 * Math.PI;   // shared floor: one jolt direction per round
                        var jolter = athletes.FirstOrDefault(a => a.In);
                        if (shake && _shared == true && jolter != null)
                        {
                            jolter.runner.Request(new Disturbance { kind = "kick", target = "shaker_x", dqvel = new[] { shakeSpeed * Math.Cos(fa) } });
                            jolter.runner.Request(new Disturbance { kind = "kick", target = "shaker_y", dqvel = new[] { shakeSpeed * Math.Sin(fa) } });
                        }
                        foreach (var a in athletes)
                        {
                            double side = _rng.NextDouble() < 0.5 ? Math.PI / 2 : -Math.PI / 2;
                            double ang = side + (_rng.NextDouble() * 2 - 1) * gustSpreadDeg * Math.PI / 180;
                            double sa = _rng.NextDouble() * 2 * Math.PI;
                            if (!a.In) continue;
                            if (a.burstAt >= 0) a.recoveries.Add(roundSeconds);
                            double bdv = dv * a.runner.Contract.SpeedScale;   // Froude: gusts scale with the body like its speeds
                            a.runner.Request(new Disturbance { kind = "shove", target = "root", dqvel = new[] { bdv * Math.Cos(ang), bdv * Math.Sin(ang), 0 } });
                            if (shake && _shared != true)
                            {
                                var p = a.ShakerPrefix;
                                a.runner.Request(new Disturbance { kind = "kick", target = p + "shaker_x", dqvel = new[] { shakeSpeed * Math.Cos(sa) } });
                                a.runner.Request(new Disturbance { kind = "kick", target = p + "shaker_y", dqvel = new[] { shakeSpeed * Math.Sin(sa) } });
                            }
                            a.burstAt = LiveTime;
                            a.calm = 0;
                        }
                        _nextRoundTick += Ticks(roundSeconds);
                    }
                    foreach (var a in athletes.Where(a => a.In))
                    {
                        var why = a.judge.Eliminated(m, d);
                        if (why != null) { a.outAt = LiveTime; a.reason = why; Stand(a); continue; }
                        if (a.burstAt < 0) continue;
                        a.calm = a.offset < calmRadius && a.speed < calmSpeed ? a.calm + Time.deltaTime : 0f;
                        if (a.calm >= calmSeconds)
                        {
                            a.recoveries.Add(Mathf.Min(roundSeconds, LiveTime - calmSeconds - a.burstAt));
                            a.burstAt = -1;
                        }
                    }
                    foreach (var a in athletes) a.total = a.recoveries.Sum() + (a.In ? 0f : roundSeconds * (rounds - a.recoveries.Count));
                    if (athletes.All(a => !a.In) || LiveTime >= startSeconds + rounds * roundSeconds) Finish();
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void Stand(Athlete a) { a.runner.steer = null; a.runner.command = Vector3.zero; }

        void Finish()
        {
            foreach (var a in athletes)
            {
                if (a.In && a.burstAt >= 0) a.recoveries.Add(roundSeconds);
                while (a.recoveries.Count < rounds) a.recoveries.Add(roundSeconds);
                a.total = a.recoveries.Sum();
                Stand(a);
            }
            int p = 1;
            foreach (var a in Standings.ToList()) a.place = p++;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = athletes.First(a => a.place == 1);
            Debug.Log($"[GustGauntlet] heat {Attempt} seed {seed + Attempt}: winner {w.name} ({Describe(w)}) after {LiveTime:F1} s");
        }

        public string Describe(Athlete a) => a.In
            ? $"{a.total:0.0} s" + (Current == Phase.Live && a.burstAt >= 0 ? $"  {a.offset * 100f:0} cm" : "")
            : $"OUT {a.outAt:0.0} s {(a.reason == "FELL" ? "fell" : "off")}";

        // IBroadcastBoard (BroadcastHud: betting window, records)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = athletes.FirstOrDefault(a => a.place == 1);
            lowerIsBetter = true;
            value = 0; text = "";
            if (Current != Phase.Result || w == null || !(true)) return false;
            value = w.total;
            text = $"{w.total:0.0} s recovery";
            return true;
        }

        public string SubtitleExtra => $"round {Round}/{rounds} · gust {GustNow:0.00} m/s";
        public string ClockLine => $"{LiveTime:0.00} s";
        public string InfoLine => $"{athletes.Count(a => a.In)} of {athletes.Count} still in";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(a => (a.place, a.name, Describe(a), !a.In, a.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{athletes.First(x => x.place == 1).name} WINS\n{Describe(athletes.First(x => x.place == 1))}",
            _ => LiveTime < 0.8f ? "HOLD!" : "",
        };
    }
}
