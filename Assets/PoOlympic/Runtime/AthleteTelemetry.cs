using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>One physics impact of an athlete, read from mjData contacts (MuJoCo world frame converted to Unity).</summary>
    public struct ImpactEvent
    {
        public enum Kind { FootStrike, BodySlam, CubeHit }
        public Kind kind;
        public PolicyRunner runner;
        public Vector3 position;        // Unity world
        public Vector3 normal;          // Unity world, pointing away from the surface that was hit
        public float forceN;            // contact normal force at the first step of the contact
        public float weightRatio;       // forceN / (athlete mass × g): 1 = one body weight
        public float speedMps;          // pelvis horizontal speed at the impact
    }

    /// <summary>
    /// Broadcast telemetry of one athlete (features 3, 5, 6, 7), read from mjData after every mj_step
    /// (MjScene.postUpdateEvent) — never from Unity transforms and never written back (parity untouched):
    ///   joint stress   |actuator_force| / force limit (strength trait included), per actuator; max + its joint
    ///   power          Σ |actuator_force · actuator_velocity| (W), smoothed
    ///   speed          pelvis horizontal speed (m/s), smoothed; heat peak
    ///   gait           foot touchdowns → cadence (steps/min) and ground contact time per step (s)
    ///   balance        centre of mass (subtree_com of the pelvis) against the support polygon = convex hull of this
    ///                  athlete's contact points on static surfaces; margin (m) &gt; 0 inside
    ///   impacts        new contacts with a surface (foot strike / body slam) or a pool cube, with the contact normal
    ///                  force (mj_contactForce) → <see cref="Impact"/>
    ///   confidence     the brain critic mapped to P(still on its feet in 2 s) (BrainConfidence)
    /// </summary>
    [DefaultExecutionOrder(-90)]
    [RequireComponent(typeof(PolicyRunner))]
    public unsafe class AthleteTelemetry : MonoBehaviour
    {
        public static event Action<ImpactEvent> Impact;
        public static readonly List<AthleteTelemetry> All = new();

        [Tooltip("Confidence model (Models/confidence_model.json, tools/fit_confidence.py).")]
        public TextAsset confidenceModel;
        [Tooltip("Seconds of confidence history kept for the HUD sparklines.")]
        public float historySeconds = 8f;

        public PolicyRunner Runner { get; private set; }
        public bool Ready { get; private set; }
        public float StressMax { get; private set; }
        public string StressJoint { get; private set; } = "";
        public float PowerW { get; private set; }
        public float PeakPowerW { get; private set; }
        public float SpeedMps { get; private set; }
        public float PeakSpeedMps { get; private set; }
        public float CadenceSpm { get; private set; }
        public float GroundContactS { get; private set; }
        public float TiltDeg { get; private set; }
        public float TiltRateDps { get; private set; }
        public float MassKg { get; private set; }
        /// <summary>Centre of mass, Unity world.</summary>
        public Vector3 Com { get; private set; }
        /// <summary>Signed distance of the CoM ground projection to the support polygon edge (m, &gt; 0 inside); NaN in the air.</summary>
        public float BalanceMargin { get; private set; } = float.NaN;
        /// <summary><see cref="BalanceMargin"/> averaged over ~0.4 s (NaN while airborne).</summary>
        public float BalanceMarginSmooth { get; private set; } = float.NaN;
        /// <summary>Support polygon (convex hull, Unity world, counter-clockwise seen from above) at <see cref="SupportHeight"/>.</summary>
        public readonly List<Vector3> SupportHull = new();
        public float SupportHeight { get; private set; }
        /// <summary>P(still on its feet in 2 s) from the brain critic; NaN when the brain has no calibrated critic.</summary>
        public float Confidence { get; private set; } = float.NaN;
        public float MinConfidence { get; private set; } = 1f;
        public readonly List<float> ConfidenceHistory = new();
        public Vector3 PelvisPosition { get; private set; }

        BrainConfidence.Entry _conf;
        float _valueEma;
        int _criticSeen;
        int _pelvis, _torso, _ngeom;
        bool[] _own, _foot, _cube;
        int[] _footIndex;                  // geom → 0..3 (foot_l, toe_l, foot_r, toe_r) or -1
        string[] _jointNames;
        readonly bool[] _footOn = new bool[2];
        readonly int[] _footOff = new int[2], _footOnSteps = new int[2];
        bool[] _bodyOnSurface, _bodyOnCube;
        int[] _bodyOffSteps, _cubeOffSteps;
        readonly List<Vector2> _pts = new();
        readonly List<float> _ptsZ = new();
        readonly Queue<float> _strikes = new();
        float _simTime, _lastTilt;
        double _dt;
        const float G = 9.81f;
        const int DebounceSteps = 12;        // 60 ms off before a contact counts as new
        const int PeakSteps = 6;             // impact force = peak over the first 30 ms of a contact

        struct Pending { public ImpactEvent.Kind kind; public Vector3 pos, normal; public float peak; public int steps; public bool touched; }
        readonly Dictionary<int, Pending> _pending = new(), _pendingNext = new();
        readonly List<int> _done = new();

        void Awake()
        {
            Runner = GetComponent<PolicyRunner>();
            var scene = MjScene.Instance;
            scene.postInitEvent += OnPostInit;
            scene.postUpdateEvent += OnPostStep;
        }

        void OnEnable() { if (!All.Contains(this)) All.Add(this); }
        void OnDisable() => All.Remove(this);

        void OnDestroy()
        {
            if (!MjScene.InstanceExists) return;
            MjScene.Instance.postInitEvent -= OnPostInit;
            MjScene.Instance.postUpdateEvent -= OnPostStep;
        }

        void OnPostInit(object sender, MjStepArgs args)
        {
            var m = args.model;
            var p = Runner.athletePrefix;
            var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
            var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);
            if (!bodies.TryGetValue(p + "pelvis", out _pelvis) || !bodies.TryGetValue(p + "torso", out _torso)) return;
            _ngeom = (int)m->ngeom;
            _own = new bool[_ngeom];
            _foot = new bool[_ngeom];
            _cube = new bool[_ngeom];
            _footIndex = new int[_ngeom];
            for (int g = 0; g < _ngeom; g++)
            {
                _own[g] = m->body_rootid[m->geom_bodyid[g]] == _pelvis;
                _cube[g] = ModelFingerprint.Name(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, g).StartsWith("cube");
                _footIndex[g] = -1;
            }
            string[] feet = { "foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0" };
            for (int i = 0; i < feet.Length; i++)
                if (geoms.TryGetValue(p + feet[i], out var g)) { _foot[g] = true; _footIndex[g] = i; }
            _bodyOnSurface = new bool[_ngeom];
            _bodyOnCube = new bool[_ngeom];
            _bodyOffSteps = new int[_ngeom];
            _cubeOffSteps = new int[_ngeom];
            for (int g = 0; g < _ngeom; g++) _bodyOffSteps[g] = _cubeOffSteps[g] = DebounceSteps;
            MassKg = (float)m->body_subtreemass[_pelvis];
            _dt = m->opt.timestep;
            var c = Runner.Contract;
            _jointNames = new string[c.actuators.Length];
            for (int i = 0; i < _jointNames.Length; i++) _jointNames[i] = c.actuators[i].joint;
            _conf = BrainConfidence.Find(confidenceModel, Runner.brain != null ? Runner.brain.name : "");
            Ready = true;
            ResetHeat();
        }

        /// <summary>New heat: clear peaks, gait and confidence history.</summary>
        public void ResetHeat()
        {
            PeakSpeedMps = PeakPowerW = 0f;
            MinConfidence = 1f;
            ConfidenceHistory.Clear();
            _strikes.Clear();
            CadenceSpm = GroundContactS = 0f;
            _criticSeen = Runner != null ? Runner.CriticEvaluations : 0;
            _valueEma = float.NaN;
        }

        void OnPostStep(object sender, MjStepArgs args)
        {
            if (!Ready || !Runner.Initialized || !isActiveAndEnabled) return;
            var m = args.model;
            var d = args.data;
            _simTime += (float)_dt;
            var ids = Runner.Binding.ActuatorIds;

            // stress + power
            float smax = 0f, power = 0f;
            int jmax = 0;
            for (int i = 0; i < ids.Length; i++)
            {
                int a = ids[i];
                double f = d->actuator_force[a];
                double lim = Math.Max(Math.Abs(m->actuator_forcerange[2 * a]), Math.Abs(m->actuator_forcerange[2 * a + 1]));
                float s = lim > 0 ? (float)(Math.Abs(f) / lim) : 0f;
                if (s > smax) { smax = s; jmax = i; }
                power += (float)Math.Abs(f * d->actuator_velocity[a]);
            }
            float k = 1f - Mathf.Exp(-(float)_dt / 0.25f);
            StressMax += (smax - StressMax) * k;
            if (smax >= StressMax) StressJoint = _jointNames[jmax];
            PowerW += (power - PowerW) * k;

            // pelvis speed, tilt
            int r = Runner.Binding.RootDofAdr;
            float sp = Mathf.Sqrt((float)(d->qvel[r] * d->qvel[r] + d->qvel[r + 1] * d->qvel[r + 1]));
            SpeedMps += (sp - SpeedMps) * (1f - Mathf.Exp(-(float)_dt / 0.3f));
            float tilt = Mathf.Acos(Mathf.Clamp((float)d->xmat[9 * _torso + 8], -1f, 1f)) * Mathf.Rad2Deg;
            float rate = (tilt - _lastTilt) / (float)_dt;
            _lastTilt = tilt;
            TiltDeg = tilt;
            TiltRateDps += (rate - TiltRateDps) * (1f - Mathf.Exp(-(float)_dt / 0.1f));
            var pel = new Vector3((float)d->xpos[3 * _pelvis], (float)d->xpos[3 * _pelvis + 2], (float)d->xpos[3 * _pelvis + 1]);
            PelvisPosition = pel;
            Com = new Vector3((float)d->subtree_com[3 * _pelvis], (float)d->subtree_com[3 * _pelvis + 2], (float)d->subtree_com[3 * _pelvis + 1]);

            // contacts: support points, foot touchdowns, new body / cube contacts (peak force over the first
            // PeakSteps of a contact: MuJoCo's soft contact force ramps up over a few steps)
            _pts.Clear();
            _ptsZ.Clear();
            bool footNowL = false, footNowR = false;
            var force = stackalloc double[6];
            for (int g = 0; g < _ngeom; g++) { _bodyOffSteps[g]++; _cubeOffSteps[g]++; }
            for (int i = 0; i < d->ncon; i++)
            {
                var c = d->contact[i];
                int g1 = c.geom1, g2 = c.geom2;
                int own, other;
                if (_own[g1] && !_own[g2]) { own = g1; other = g2; }
                else if (_own[g2] && !_own[g1]) { own = g2; other = g1; }
                else continue;
                double px = c.pos[0], py = c.pos[1], pz = c.pos[2];
                bool cube = _cube[other];
                if (!cube)
                {
                    _pts.Add(new Vector2((float)px, (float)py));
                    _ptsZ.Add((float)pz);
                    if (_foot[own])
                    {
                        if (_footIndex[own] < 2) footNowL = true; else footNowR = true;
                        continue;
                    }
                }
                bool fresh = cube ? _cubeOffSteps[own] >= DebounceSteps && !_bodyOnCube[own]
                                  : _bodyOffSteps[own] >= DebounceSteps && !_bodyOnSurface[own];
                if (cube) { _bodyOnCube[own] = true; _cubeOffSteps[own] = 0; }
                else { _bodyOnSurface[own] = true; _bodyOffSteps[own] = 0; }
                bool tracking = _pending.TryGetValue(own, out var pend);
                if (!fresh && !tracking) continue;
                MujocoLib.mj_contactForce(m, d, i, force);
                float fn = (float)Math.Abs(force[0]);
                if (!tracking)
                {
                    // contact frame x axis = normal, pointing from geom1 to geom2 → away from the other surface
                    float sign = own == g2 ? 1f : -1f;
                    pend = new Pending
                    {
                        kind = cube ? ImpactEvent.Kind.CubeHit : ImpactEvent.Kind.BodySlam,
                        pos = new Vector3((float)px, (float)pz, (float)py),
                        normal = new Vector3((float)c.frame[0], (float)c.frame[2], (float)c.frame[1]) * sign,
                    };
                }
                pend.peak = Mathf.Max(pend.peak, fn);
                pend.touched = true;
                _pending[own] = pend;
            }
            for (int g = 0; g < _ngeom; g++)
            {
                if (_bodyOffSteps[g] > 0) _bodyOnSurface[g] = false;
                if (_cubeOffSteps[g] > 0) _bodyOnCube[g] = false;
            }
            if (_pending.Count > 0)
            {
                _done.Clear();
                foreach (var kv in _pending)
                {
                    var pend = kv.Value;
                    pend.steps++;
                    if (pend.steps >= PeakSteps || !pend.touched) _done.Add(kv.Key);
                    pend.touched = false;
                    _pendingNext[kv.Key] = pend;
                }
                foreach (var kv in _pendingNext) _pending[kv.Key] = kv.Value;
                _pendingNext.Clear();
                foreach (var g in _done)
                {
                    var pend = _pending[g];
                    _pending.Remove(g);
                    Impact?.Invoke(new ImpactEvent
                    {
                        kind = pend.kind, runner = Runner, position = pend.pos, normal = pend.normal, forceN = pend.peak,
                        weightRatio = pend.peak / Mathf.Max(1f, MassKg * G), speedMps = SpeedMps,
                    });
                }
            }
            Foot(0, footNowL, d);
            Foot(1, footNowR, d);
            while (_strikes.Count > 0 && _simTime - _strikes.Peek() > 3f) _strikes.Dequeue();
            CadenceSpm = _strikes.Count >= 2 ? (_strikes.Count - 1) * 60f / Mathf.Max(0.3f, _simTime - _strikes.Peek()) : 0f;
            if (!float.IsNaN(Com.x)) UpdateBalance();

            // heat peaks (after the 1 s smoothing)
            PeakSpeedMps = Mathf.Max(PeakSpeedMps, SpeedMps);
            PeakPowerW = Mathf.Max(PeakPowerW, PowerW);
            UpdateConfidence();
        }

        void Foot(int side, bool on, MujocoLib.mjData_* d)
        {
            if (on)
            {
                if (!_footOn[side] && _footOff[side] >= DebounceSteps)
                {
                    _strikes.Enqueue(_simTime);
                    // touchdown point: the latest foot contact of this side
                    Vector3 pos = PelvisPosition; pos.y = 0f;
                    if (_pts.Count > 0) { var q = _pts[_pts.Count - 1]; pos = new Vector3(q.x, _ptsZ[_ptsZ.Count - 1], q.y); }
                    Impact?.Invoke(new ImpactEvent
                    {
                        kind = ImpactEvent.Kind.FootStrike, runner = Runner, position = pos, normal = Vector3.up,
                        forceN = MassKg * G * (1f + SpeedMps * 0.35f), weightRatio = 1f + SpeedMps * 0.35f, speedMps = SpeedMps,
                    });
                }
                _footOn[side] = true;
                _footOnSteps[side]++;
                _footOff[side] = 0;
            }
            else
            {
                if (_footOn[side] && _footOnSteps[side] > 2)
                    GroundContactS += (_footOnSteps[side] * (float)_dt - GroundContactS) * 0.3f;   // stance time per step, smoothed
                _footOn[side] = false;
                _footOnSteps[side] = 0;
                _footOff[side]++;
            }
        }

        void UpdateBalance()
        {
            SupportHull.Clear();
            if (_pts.Count == 0) { BalanceMargin = BalanceMarginSmooth = float.NaN; return; }
            float z = 0f;
            foreach (var h in _ptsZ) z += h;
            SupportHeight = z / _ptsZ.Count;
            var hull = Geometry2D.ConvexHull(_pts);
            var com = new Vector2(Com.x, Com.z);
            BalanceMargin = Geometry2D.SignedDistance(hull, com);
            BalanceMarginSmooth = float.IsNaN(BalanceMarginSmooth) ? BalanceMargin
                : BalanceMarginSmooth + (BalanceMargin - BalanceMarginSmooth) * (1f - Mathf.Exp(-(float)_dt / 0.4f));
            foreach (var q in hull) SupportHull.Add(new Vector3(q.x, SupportHeight, q.y));
        }

        void UpdateConfidence()
        {
            if (_conf == null || Runner.CriticEvaluations == _criticSeen) return;
            _criticSeen = Runner.CriticEvaluations;
            float v = Runner.CriticValue;
            float alpha = Mathf.Clamp01(Runner.criticEvery * (float)(_dt * Runner.Contract.decimation) / _conf.emaTau);
            _valueEma = float.IsNaN(_valueEma) ? v : _valueEma + alpha * (v - _valueEma);
            var cmd = Runner.command;
            Confidence = _conf.Probability(v, v - _valueEma, Mathf.Sqrt(cmd.x * cmd.x + cmd.y * cmd.y));
            MinConfidence = Mathf.Min(MinConfidence, Confidence);
            ConfidenceHistory.Add(Confidence);
            int keep = Mathf.CeilToInt(historySeconds / (Runner.criticEvery * (float)(_dt * Runner.Contract.decimation)));
            while (ConfidenceHistory.Count > keep) ConfidenceHistory.RemoveAt(0);
        }
    }

    /// <summary>The critic → probability map fitted by tools/fit_confidence.py:
    /// P = sigmoid(w_value·V + w_drop·(V − EMA(V)) + w_speed·|cmd_xy| + b).</summary>
    public static class BrainConfidence
    {
        [Serializable]
        public class Entry
        {
            public string brain;
            public float w_value, w_drop, w_speed, b, auc;
            public bool calibrated;
            [NonSerialized] public float emaTau = 1f;
            public float Probability(float value, float drop, float speed) =>
                1f / (1f + Mathf.Exp(-(w_value * value + w_drop * drop + w_speed * speed + b)));
        }

        [Serializable] class Model { public float horizon_s; public float ema_tau_s = 1f; public Entry[] brains; }

        static readonly Dictionary<TextAsset, Model> Cache = new();

        /// <summary>The calibrated entry for a brain (ModelAsset name = file name without .onnx), else null.</summary>
        public static Entry Find(TextAsset json, string brain)
        {
            if (json == null || string.IsNullOrEmpty(brain)) return null;
            if (!Cache.TryGetValue(json, out var model)) Cache[json] = model = JsonUtility.FromJson<Model>(json.text);
            if (model?.brains == null) return null;
            foreach (var e in model.brains)
                if (e.brain == brain && e.calibrated) { e.emaTau = model.ema_tau_s > 0 ? model.ema_tau_s : 1f; return e; }
            return null;
        }
    }

    /// <summary>Convex hull (monotone chain) and signed point-to-polygon distance on the ground plane.</summary>
    public static class Geometry2D
    {
        static float Cross(Vector2 o, Vector2 a, Vector2 b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

        /// <summary>Counter-clockwise hull; 1 or 2 points for degenerate input (a point / a segment).</summary>
        public static List<Vector2> ConvexHull(List<Vector2> pts)
        {
            var p = new List<Vector2>(pts);
            p.Sort((a, b) => a.x != b.x ? a.x.CompareTo(b.x) : a.y.CompareTo(b.y));
            var dedup = new List<Vector2>();
            foreach (var q in p)
                if (dedup.Count == 0 || (q - dedup[dedup.Count - 1]).sqrMagnitude > 1e-8f) dedup.Add(q);
            if (dedup.Count < 3) return dedup;
            var h = new List<Vector2>();
            foreach (var q in dedup)
            {
                while (h.Count >= 2 && Cross(h[h.Count - 2], h[h.Count - 1], q) <= 0) h.RemoveAt(h.Count - 1);
                h.Add(q);
            }
            int lower = h.Count + 1;
            for (int i = dedup.Count - 2; i >= 0; i--)
            {
                var q = dedup[i];
                while (h.Count >= lower && Cross(h[h.Count - 2], h[h.Count - 1], q) <= 0) h.RemoveAt(h.Count - 1);
                h.Add(q);
            }
            h.RemoveAt(h.Count - 1);
            return h;
        }

        static float SegmentDistance(Vector2 p, Vector2 a, Vector2 b)
        {
            var ab = b - a;
            float t = ab.sqrMagnitude > 1e-12f ? Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude) : 0f;
            return (a + t * ab - p).magnitude;
        }

        /// <summary>Distance to the hull boundary, positive inside (point / segment hulls: always ≤ 0).</summary>
        public static float SignedDistance(List<Vector2> hull, Vector2 p)
        {
            if (hull.Count == 0) return float.NaN;
            if (hull.Count == 1) return -(hull[0] - p).magnitude;
            float best = float.MaxValue;
            bool inside = hull.Count >= 3;
            for (int i = 0; i < hull.Count; i++)
            {
                var a = hull[i];
                var b = hull[(i + 1) % hull.Count];
                best = Mathf.Min(best, SegmentDistance(p, a, b));
                if (hull.Count >= 3 && Cross(a, b, p) < 0) inside = false;
            }
            return inside ? best : -best;
        }
    }
}
