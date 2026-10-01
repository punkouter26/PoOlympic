using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 6 The Flamingo Classic — 8 athletes stand on one leg on the venue's 2 x 4 station grid, Rung S brain (contract v4
    /// lift-foot command: stand on the other leg). Mirror of training/poolympic/events/flamingo.py: at GO everyone lifts
    /// the same foot (seeded per heat) and has `liftSeconds` to get it off the ground; the stance foot's spot is marked
    /// then. From there the wind rises: every `roundSeconds` each athlete still up gets a gust of the same magnitude
    /// (gustStart + gustStep per round) in its own seeded direction.
    ///   out = the lifted foot touching the ground (TOUCHDOWN), the stance foot more than `hopTol` from its spot
    ///         (HOPPED), or a fall (FELL)
    /// The mark is the time on one leg. Rank by it, longest first; the heat runs until the last flamingo is down (anyone
    /// still up at `maxSeconds` shares first place).
    ///   Ready (countdown) → Live → Result → auto restart (new seed, new traits)
    /// </summary>
    public class FlamingoEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }

        [Serializable]
        public class Flamingo
        {
            public PolicyRunner runner;
            public string name;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float outAt = -1, hop;
            [NonSerialized] public int place, gusts, footL = -1, footR = -1;
            [NonSerialized] public double spotX, spotY;
            public bool In => status.Length == 0;
        }

        public List<Flamingo> flamingos = new();

        public void DropInactiveLanes() => flamingos.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);

        [Header("Rules (= flamingo.py)")]
        public float liftSeconds = 2f;
        public float roundSeconds = 2f;
        public float gustStart = 0.04f, gustStep = 0.012f;
        public float hopTol = 0.30f;
        public float maxSeconds = 120f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public int Round { get; private set; }
        public SkillCommand.Foot Lifted { get; private set; }     // the foot in the air this heat
        public float GustNow => Round == 0 ? 0f : gustStart + gustStep * (Round - 1);
        public int StillUp => flamingos.Count(s => s.In);

        public IEnumerable<Flamingo> Standings => Current == Phase.Result
            ? flamingos.OrderBy(s => s.place)
            : flamingos.OrderBy(s => s.In ? 0 : 1).ThenByDescending(s => s.outAt);

        System.Random _rng;
        int _liveStartTick, _nextRoundTick;
        bool _traitsPending = true, _marked;

        float TickSeconds => (float)(flamingos[0].runner.Contract.timestep * flamingos[0].runner.Contract.decimation);
        int Ticks(float s) => Mathf.RoundToInt(s / TickSeconds);

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var s in flamingos)
            {
                s.status = ""; s.outAt = -1; s.hop = 0; s.place = s.gusts = 0;
                s.runner.command = Vector3.zero;
                s.runner.laneKeeping = false;
                s.runner.skill = default;
                if (!first) s.runner.RequestReset();
            }
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            Round = 0;
            Lifted = SkillCommand.Foot.None;
            _marked = false;
            _traitsPending = true;
        }

        void DrawTraits()
        {
            var tr = flamingos[0].runner.Contract.trait_ranges;
            for (int i = 0; i < flamingos.Count; i++)
            {
                var r = flamingos[i].runner;
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
            if (flamingos.Count == 0 || flamingos.Any(s => s.runner == null || !s.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var s in flamingos)
            {
                if (s.judge != null) continue;
                s.judge = new AthleteJudge(m, s.runner, "ground");
                var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
                s.footL = bodies[s.runner.athletePrefix + "foot_l"];
                s.footR = bodies[s.runner.athletePrefix + "foot_r"];
            }
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = flamingos[0].runner;
            switch (Current)
            {
                case Phase.Ready:
                    if (HoldStart) { PhaseTime = 0; break; }   // betting window (BroadcastHud)
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = lead.ControlTick;
                        _nextRoundTick = _liveStartTick + Ticks(liftSeconds + roundSeconds);
                        Lifted = _rng.Next(2) == 0 ? SkillCommand.Foot.Left : SkillCommand.Foot.Right;
                        foreach (var s in flamingos) s.runner.skill.liftFoot = Lifted;
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * TickSeconds;
                    bool judged = lead.ControlTick >= _liveStartTick + Ticks(liftSeconds);
                    bool left = Lifted == SkillCommand.Foot.Left;
                    if (judged && !_marked)                    // the stance foot's spot, marked when the lift time is up
                    {
                        _marked = true;
                        foreach (var s in flamingos)
                        {
                            int b = left ? s.footR : s.footL;
                            s.spotX = d->xpos[3 * b];
                            s.spotY = d->xpos[3 * b + 1];
                        }
                    }
                    if (lead.ControlTick >= _nextRoundTick)    // gust round: same magnitude, own direction
                    {
                        Round++;
                        double dv = GustNow;
                        foreach (var s in flamingos.Where(s => s.In))
                        {
                            double a = _rng.NextDouble() * 2 * Math.PI;
                            s.runner.Request(new Disturbance { kind = "shove", target = "root", dqvel = new[] { dv * Math.Cos(a), dv * Math.Sin(a), 0 } });
                            s.gusts++;
                        }
                        _nextRoundTick += Ticks(roundSeconds);
                    }
                    foreach (var s in flamingos.Where(s => s.In))
                    {
                        string why = s.judge.Eliminated(m, d) != null ? "FELL" : null;
                        if (why == null && judged)
                        {
                            int b = left ? s.footR : s.footL;
                            double dx = d->xpos[3 * b] - s.spotX, dy = d->xpos[3 * b + 1] - s.spotY;
                            s.hop = Mathf.Max(s.hop, (float)Math.Sqrt(dx * dx + dy * dy));
                            why = s.judge.FootDown(d, left ? 0 : 1) ? "TOUCHDOWN" : s.hop > hopTol ? "HOPPED" : null;
                        }
                        if (why != null) Out(s, why);
                    }
                    if (StillUp == 0 || LiveTime >= maxSeconds) Finish();
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void Out(Flamingo s, string status)
        {
            s.status = status;
            s.outAt = LiveTime;
            s.runner.skill.liftFoot = SkillCommand.Foot.None;
        }

        /// <summary>Time on one leg: the exit time, or the heat's length for a flamingo still up at `maxSeconds`.</summary>
        public float Mark(Flamingo s) => s.outAt < 0 ? LiveTime : s.outAt;

        void Finish()
        {
            var order = flamingos.OrderBy(s => s.In ? 0 : 1).ThenByDescending(s => s.outAt).ToList();
            int place = 1;
            for (int i = 0; i < order.Count; i++)
            {
                bool tie = i > 0 && ((order[i].In && order[i - 1].In) || (!order[i].In && !order[i - 1].In && Mathf.Approximately(order[i].outAt, order[i - 1].outAt)));
                if (!tie) place = i + 1;
                order[i].place = place;
            }
            foreach (var s in flamingos) s.runner.skill.liftFoot = SkillCommand.Foot.None;
            Current = Phase.Result;
            PhaseTime = 0;
            var w = order[0];
            Debug.Log($"[Flamingo] heat {Attempt} seed {seed + Attempt}: {Lifted} foot up, winner {w.name} {Mark(w):0.0} s " +
                      $"({(w.In ? "UP" : w.status)}), {Round} gusts up to {GustNow:0.00} m/s");
        }

        public string Describe(Flamingo s) => s.status switch
        {
            _ when Current == Phase.Result && s.place == 1 => $"{Mark(s):0.0} s",   // the winner's mark, not how it ended
            "" => "UP",
            "TOUCHDOWN" => $"DOWN {s.outAt:0.0} s",
            _ => $"{s.status} {s.outAt:0.0} s",
        };

        // IBroadcastBoard (BroadcastHud: betting window, records)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        /// <summary>Record = the winner's time on one leg.</summary>
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = flamingos.FirstOrDefault(s => s.place == 1);
            lowerIsBetter = false;
            value = 0; text = "";
            if (Current != Phase.Result || w == null) return false;
            value = Mark(w);
            text = $"{Mark(w):0.0} s on one leg";
            return true;
        }

        public string SubtitleExtra => "last one up wins";
        // short lines: the clock sits between the FPS pill and the menu button
        public string ClockLine => $"{LiveTime:0.0} s";
        public string InfoLine => Current == Phase.Ready ? "one leg, rising gusts"
            : Round == 0 ? $"{StillUp} of {flamingos.Count} up"
            : $"{StillUp} of {flamingos.Count} up · gust {GustNow:0.00} m/s";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(s => (s.place, s.name, Describe(s), !s.In && !(Current == Phase.Result && s.place == 1), s.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{string.Join(" & ", flamingos.Where(x => x.place == 1).Select(x => x.name))} WINS\n{Mark(flamingos.First(x => x.place == 1)):0.0} s",
            _ => LiveTime < 0.8f ? (Lifted == SkillCommand.Foot.Left ? "LEFT FOOT UP!" : "RIGHT FOOT UP!") : "",
        };
    }
}
