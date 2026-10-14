using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 2 Torso Archer and 4 Precision Javelin Reach — 8 athletes on the venue's 2 x 4 station grid (scene_archer8 /
    /// reach8_mattbio.xml, no contact), Rung S brains (contract v4 torso-aim / hand-target commands). Mirror of
    /// training/poolympic/events/targets.py: `rounds` timed targets that get faster (round r lasts
    /// max(tMin, tStart − tStep·r) s), the same targets for every lane.
    ///   Archer: the aim (chest yaw, pitch) flies in a straight line to a new aim over the first `moveFrac` of the round,
    ///           then hangs; error = angle off the target, averaged over the whole round.
    ///   Reach:  a target at the reaching arm's side (arms alternate), farther out every round (reachFrac of the reach
    ///           radius); error = distance forearm tip − target over the last `sampleFrac` of the round.
    /// Round points = roundPoints × clamp(1 − error / tol, 0, 1). Out = a fall or a foot sliding more than `stepTol`
    /// (STEPPED). Finishing every round in balance earns `balanceBonus`. Rank by points; ties: the later exit first.
    ///   Ready (countdown) → Live → Result → auto restart (new seed)
    /// </summary>
    public class TargetEvent : MonoBehaviour, IBroadcastBoard, ILaneRoster
    {
        public enum Phase { Ready, Live, Result }
        public enum Kind { Archer, Reach }

        [Serializable]
        public class Aimer
        {
            public PolicyRunner runner;
            public string name;
            [Tooltip("Shows this lane's target (built by EventScenes; moved, never created, at runtime).")]
            public Transform marker;
            [NonSerialized] public AthleteJudge judge;
            [NonSerialized] public string status = "";
            [NonSerialized] public float points, outAt = -1, lastErr = -1, errSum;
            [NonSerialized] public int errN, place, pelvis = -1, chest, footL, footR;
            [NonSerialized] public readonly int[] forearm = new int[2], upperArm = new int[2];
            [NonSerialized] public readonly Vector3[] tipLocal = new Vector3[2], shoulder = new Vector3[2];   // MuJoCo axes
            [NonSerialized] public double fallTilt;
            [NonSerialized] public Vector2 startL, startR;
            public bool In => status.Length == 0;
        }

        public Kind kind;
        public List<Aimer> aimers = new();

        public void DropInactiveLanes() => aimers.RemoveAll(x => x.runner == null || !x.runner.gameObject.activeInHierarchy);

        [Header("Rules (= targets.py)")]
        public int rounds = 12;
        public float tStart = 2.0f, tStep = 0.1f, tMin = 1.0f;
        public float roundPoints = 10f, balanceBonus = 10f, stepTol = 0.20f;
        [Header("Archer")]
        public float archerYawDeg = 45f;
        public Vector2 archerPitchDeg = new(-15f, 35f);
        public float archerMinMoveDeg = 30f, moveFrac = 0.6f, archerTol = 0.25f;
        [Tooltip("Distance of the flight target from the chest (display only).")]
        public float archerMarkerRange = 2.5f;
        [Header("Reach")]
        [Tooltip("Reach radius from the shoulder in MATT units (contract.py SKILL_RANGES hand_reach).")]
        public float handReach = 0.75f;
        public Vector2 reachFrac = new(0.50f, 0.78f), reachAzDeg = new(-30f, 90f), reachElDeg = new(-40f, 45f);
        public float sampleFrac = 0.4f, reachTol = 0.15f;
        public float countdownSeconds = 3f, resultHoldSeconds = 6f;
        public bool autoRestart = true, randomTraits = true;
        public int seed = 1;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public int Round { get; private set; } = -1;
        public float MaxPoints => rounds * roundPoints + balanceBonus;
        public float RoundS(int r) => Mathf.Max(tMin, tStart - tStep * r);
        float Tol => kind == Kind.Archer ? archerTol : reachTol;

        public IEnumerable<Aimer> Standings => Current == Phase.Result
            ? aimers.OrderBy(s => s.place)
            : aimers.OrderByDescending(s => s.points).ThenBy(s => s.In ? 0 : 1).ThenByDescending(s => s.outAt);

        System.Random _rng;
        int _liveStartTick;
        bool _traitsPending = true;
        Vector2[] _aims;                                   // archer: rounds + 1 aims (yaw, pitch) in rad
        (int arm, float az, float el, float frac)[] _dirs;  // reach: per round

        void Start() => BeginAttempt(first: true);
        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            foreach (var s in aimers)
            {
                s.status = ""; s.points = 0; s.outAt = s.lastErr = -1; s.errSum = 0; s.errN = 0; s.place = 0;
                s.runner.command = Vector3.zero;
                s.runner.laneKeeping = false;
                s.runner.skill = default;
                if (s.marker != null) s.marker.gameObject.SetActive(false);
                if (!first) s.runner.RequestReset();
            }
            DrawSchedule(new System.Random(seed * 7919 + Attempt));
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            Round = -1;
            _traitsPending = true;
        }

        void DrawSchedule(System.Random rng)
        {
            float U(float lo, float hi) => lo + (hi - lo) * (float)rng.NextDouble();
            _aims = new Vector2[rounds + 1];
            for (int r = 1; r <= rounds; r++)
                do _aims[r] = new Vector2(U(-archerYawDeg, archerYawDeg), U(archerPitchDeg.x, archerPitchDeg.y)) * Mathf.Deg2Rad;
                while ((_aims[r] - _aims[r - 1]).magnitude < archerMinMoveDeg * Mathf.Deg2Rad);
            int firstArm = rng.NextDouble() < 0.5 ? -1 : 1;
            _dirs = new (int, float, float, float)[rounds];
            for (int r = 0; r < rounds; r++)
                _dirs[r] = (r % 2 == 0 ? firstArm : -firstArm, U(reachAzDeg.x, reachAzDeg.y) * Mathf.Deg2Rad,
                            U(reachElDeg.x, reachElDeg.y) * Mathf.Deg2Rad,
                            Mathf.Lerp(reachFrac.x, reachFrac.y, rounds > 1 ? r / (rounds - 1f) : 1f));
        }

        void DrawTraits()
        {
            var tr = aimers[0].runner.Contract.trait_ranges;
            for (int i = 0; i < aimers.Count; i++)
                aimers[i].runner.DrawTraits(tr, _rng, randomTraits, seed * 1000 + Attempt * 16 + i);
        }

        /// <summary>Round at `live` s after GO (-1 before GO and after the last round) and how far through it (0..1).</summary>
        int RoundAt(float live, out float frac)
        {
            frac = 0;
            if (live < 0) return -1;
            float t = 0;
            for (int r = 0; r < rounds; r++)
            {
                if (live < t + RoundS(r) - 1e-4f) { frac = (live - t) / RoundS(r); return r; }
                t += RoundS(r);
            }
            return -1;
        }

        /// <summary>Target aim (yaw, pitch) `frac` through round r.</summary>
        public Vector2 ArcherTarget(int r, float frac) => Vector2.Lerp(_aims[r], _aims[r + 1], Mathf.Clamp01(frac / moveFrac));

        /// <summary>Hand target of round r in the heading frame relative to the pelvis (MuJoCo axes).</summary>
        Vector3 ReachTarget(Aimer s, int r)
        {
            var (arm, az, el, frac) = _dirs[r];
            var u = new Vector3(Mathf.Cos(el) * Mathf.Cos(az), -arm * Mathf.Cos(el) * Mathf.Sin(az), Mathf.Sin(el));
            return s.shoulder[arm < 0 ? 0 : 1] + u * (handReach * frac);
        }

        unsafe void Bind(MujocoLib.mjModel_* m, Aimer s)
        {
            s.judge = new AthleteJudge(m, s.runner, "ground");
            s.fallTilt = s.judge.fallTiltDeg;
            var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
            string p = s.runner.athletePrefix;
            s.pelvis = bodies[p + "pelvis"];
            s.chest = bodies[p + "chest"];
            s.footL = bodies[p + "foot_l"];
            s.footR = bodies[p + "foot_r"];
            for (int k = 0; k < 2; k++)
            {
                string side = k == 0 ? "_l" : "_r";
                s.forearm[k] = bodies[p + "forearm" + side];
                s.upperArm[k] = bodies[p + "upper_arm" + side];
                // skills.forearm_tip_local: the far end of the forearm capsule (+ radius) from the elbow, body frame
                int g = 0;
                while (m->geom_bodyid[g] != s.forearm[k]) g++;
                double w = m->geom_quat[4 * g], x = m->geom_quat[4 * g + 1], y = m->geom_quat[4 * g + 2], z = m->geom_quat[4 * g + 3];
                var axis = new Vector3((float)(2 * (x * z + w * y)), (float)(2 * (y * z - w * x)), (float)(1 - 2 * (x * x + y * y)));
                var pos = new Vector3((float)m->geom_pos[3 * g], (float)m->geom_pos[3 * g + 1], (float)m->geom_pos[3 * g + 2]);
                float radius = (float)m->geom_size[3 * g], half = (float)m->geom_size[3 * g + 1];
                Vector3 a = pos + axis * half, b = pos - axis * half;
                var far = a.sqrMagnitude >= b.sqrMagnitude ? a : b;
                s.tipLocal[k] = far + radius * far.normalized;
            }
        }

        static unsafe Vector3 Pos(MujocoLib.mjData_* d, int body) =>
            new((float)d->xpos[3 * body], (float)d->xpos[3 * body + 1], (float)d->xpos[3 * body + 2]);

        static unsafe float Yaw(MujocoLib.mjData_* d, int body) => Mathf.Atan2((float)d->xmat[9 * body + 3], (float)d->xmat[9 * body]);

        /// <summary>World offset (MuJoCo axes) → the pelvis heading frame (x forward, y left, z up).</summary>
        static Vector3 ToHeading(Vector3 rel, float yaw)
        {
            float c = Mathf.Cos(yaw), s = Mathf.Sin(yaw);
            return new Vector3(c * rel.x + s * rel.y, -s * rel.x + c * rel.y, rel.z);
        }

        static Vector3 FromHeading(Vector3 v, float yaw)
        {
            float c = Mathf.Cos(yaw), s = Mathf.Sin(yaw);
            return new Vector3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
        }

        static Vector3 ToUnity(Vector3 mj) => new(mj.x, mj.z, mj.y);

        /// <summary>skills.torso_aim: chest x-axis yaw relative to the pelvis heading, pitch = forward lean.</summary>
        static unsafe Vector2 TorsoAim(MujocoLib.mjData_* d, Aimer s)
        {
            double x0 = d->xmat[9 * s.chest], x1 = d->xmat[9 * s.chest + 3], x2 = d->xmat[9 * s.chest + 6];
            float yaw = Mathf.DeltaAngle(Yaw(d, s.pelvis) * Mathf.Rad2Deg, Mathf.Atan2((float)x1, (float)x0) * Mathf.Rad2Deg) * Mathf.Deg2Rad;
            return new Vector2(yaw, Mathf.Asin(Mathf.Clamp((float)-x2, -1f, 1f)));
        }

        /// <summary>skills.hand_in_heading: forearm tip of arm k (0 left, 1 right) in the pelvis heading frame.</summary>
        static unsafe Vector3 Hand(MujocoLib.mjData_* d, Aimer s, int k)
        {
            int b = s.forearm[k];
            var t = s.tipLocal[k];
            var tip = Pos(d, b) + new Vector3(
                (float)(d->xmat[9 * b] * t.x + d->xmat[9 * b + 1] * t.y + d->xmat[9 * b + 2] * t.z),
                (float)(d->xmat[9 * b + 3] * t.x + d->xmat[9 * b + 4] * t.y + d->xmat[9 * b + 5] * t.z),
                (float)(d->xmat[9 * b + 6] * t.x + d->xmat[9 * b + 7] * t.y + d->xmat[9 * b + 8] * t.z));
            return ToHeading(tip - Pos(d, s.pelvis), Yaw(d, s.pelvis));
        }

        unsafe void Update()
        {
            if (aimers.Count == 0 || aimers.Any(s => s.runner == null || !s.runner.Initialized)) return;
            if (!MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            foreach (var s in aimers)
                if (s.pelvis < 0) Bind(m, s);
            if (_traitsPending) { DrawTraits(); _traitsPending = false; }
            PhaseTime += Time.deltaTime;
            var lead = aimers[0].runner;
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
                        foreach (var s in aimers)
                        {
                            s.startL = FootXY(d, s.footL);
                            s.startR = FootXY(d, s.footR);
                            float yaw = Yaw(d, s.pelvis);
                            for (int k = 0; k < 2; k++) s.shoulder[k] = ToHeading(Pos(d, s.upperArm[k]) - Pos(d, s.pelvis), yaw);
                        }
                    }
                    break;
                case Phase.Live:
                    LiveTime = (lead.ControlTick - _liveStartTick) * tick;
                    int round = RoundAt(LiveTime, out float frac);
                    if (round != Round)
                    {
                        if (Round >= 0) ScoreRound();
                        Round = round;
                    }
                    if (round < 0) { Finish(); break; }
                    foreach (var s in aimers.Where(s => s.In))
                    {
                        float yaw = Yaw(d, s.pelvis);
                        var pelvis = Pos(d, s.pelvis);
                        float err;
                        Vector3 mark;
                        if (kind == Kind.Archer)
                        {
                            var aim = ArcherTarget(round, frac);
                            s.runner.skill = new SkillCommand { torsoYaw = aim.x, torsoPitch = aim.y };
                            s.judge.fallTiltDeg = s.fallTilt + Mathf.Max(0f, aim.y) * Mathf.Rad2Deg;
                            var got = TorsoAim(d, s);
                            err = new Vector2(Mathf.DeltaAngle(aim.x * Mathf.Rad2Deg, got.x * Mathf.Rad2Deg) * Mathf.Deg2Rad, got.y - aim.y).magnitude;
                            var u = new Vector3(Mathf.Cos(aim.y) * Mathf.Cos(aim.x), Mathf.Cos(aim.y) * Mathf.Sin(aim.x), -Mathf.Sin(aim.y));
                            mark = Pos(d, s.chest) + FromHeading(u, yaw) * archerMarkerRange;
                        }
                        else
                        {
                            var target = ReachTarget(s, round);
                            int arm = _dirs[round].arm;
                            s.runner.skill = new SkillCommand { handTarget = target, arm = arm };
                            err = frac > 1f - sampleFrac ? (Hand(d, s, arm < 0 ? 0 : 1) - target).magnitude : -1f;
                            mark = pelvis + FromHeading(target, yaw);
                        }
                        if (s.marker != null)
                        {
                            s.marker.position = ToUnity(mark);
                            if (!s.marker.gameObject.activeSelf) s.marker.gameObject.SetActive(true);
                        }
                        string why = s.judge.Eliminated(m, d);
                        if (why == null && (Vector2.Distance(FootXY(d, s.footL), s.startL) > stepTol ||
                                            Vector2.Distance(FootXY(d, s.footR), s.startR) > stepTol)) why = "STEPPED";
                        if (why != null) { Out(s, why == "STEPPED" ? why : "FELL"); continue; }
                        if (err >= 0f) { s.errSum += err; s.errN++; }
                    }
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        static unsafe Vector2 FootXY(MujocoLib.mjData_* d, int body) => new((float)d->xpos[3 * body], (float)d->xpos[3 * body + 1]);

        void ScoreRound()
        {
            foreach (var s in aimers)
            {
                if (s.In && s.errN > 0)
                {
                    s.lastErr = s.errSum / s.errN;
                    s.points += roundPoints * Mathf.Clamp01(1f - s.lastErr / Tol);
                }
                s.errSum = 0; s.errN = 0;
            }
        }

        void Out(Aimer s, string status)
        {
            s.status = status;
            s.outAt = LiveTime;
            s.runner.skill = default;
            if (s.marker != null) s.marker.gameObject.SetActive(false);
        }

        void Finish()
        {
            foreach (var s in aimers.Where(s => s.In)) { s.status = "DONE"; s.points += balanceBonus; }
            int p = 1;
            foreach (var s in aimers.OrderByDescending(s => s.points).ThenByDescending(s => s.outAt < 0 ? 1e9f : s.outAt)) s.place = p++;
            foreach (var s in aimers)
            {
                s.runner.skill = default;
                if (s.marker != null) s.marker.gameObject.SetActive(false);
            }
            Current = Phase.Result;
            PhaseTime = 0;
            var w = aimers.First(s => s.place == 1);
            Debug.Log($"[TargetEvent] {kind} heat {Attempt} seed {seed + Attempt}: winner {w.name} {w.points:0.0} pts ({w.status}) after {LiveTime:F1} s");
        }

        string ErrText(float err) => kind == Kind.Archer ? $"{err * Mathf.Rad2Deg:0}°" : $"{err * 100f:0} cm";

        public string Describe(Aimer s) => s.status switch
        {
            "" => $"{s.points:0.0} pts" + (s.lastErr >= 0 ? $"  {ErrText(s.lastErr)}" : ""),
            "DONE" => $"{s.points:0.0} pts",
            _ => $"{s.points:0.0} pts {s.status} {s.outAt:0.0} s",
        };

        // IBroadcastBoard (BroadcastHud: betting window, records)
        public BoardPhase BoardState => (BoardPhase)(int)Current;
        public int Heat => Attempt;
        public bool HoldStart { get; set; }
        public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
        {
            var w = aimers.FirstOrDefault(s => s.place == 1);
            lowerIsBetter = false;
            value = 0; text = "";
            if (Current != Phase.Result || w == null) return false;
            value = w.points;
            text = $"{w.points:0.0} pts";
            return true;
        }

        public string SubtitleExtra => kind == Kind.Archer ? $"{rounds} flights, ±{archerYawDeg:0}°" : $"{rounds} targets, out to {reachFrac.y * handReach * 100f:0} cm";
        public string ClockLine => Round < 0 ? $"{LiveTime:0.0} s" : $"{(kind == Kind.Archer ? "FLIGHT" : "TARGET")} {Round + 1}/{rounds}";
        public string InfoLine => Round < 0 ? $"max {MaxPoints:0} pts"
            : kind == Kind.Archer ? $"{RoundS(Round):0.0} s per flight" : $"{(_dirs[Round].arm < 0 ? "LEFT" : "RIGHT")} arm  {_dirs[Round].frac * handReach * 100f:0} cm";
        public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
            Standings.Select(s => (s.place, s.name, Describe(s), s.status is "FELL" or "STEPPED", s.runner));
        public string Banner => Current switch
        {
            Phase.Ready => Mathf.CeilToInt(countdownSeconds - PhaseTime).ToString(),
            Phase.Result => $"{aimers.First(x => x.place == 1).name} WINS\n{aimers.First(x => x.place == 1).points:0.0} pts",
            _ => LiveTime < 0.8f ? (kind == Kind.Archer ? "TRACK!" : "REACH!") : "",
        };
    }
}
