using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Drives one athlete from an ONNX brain at the contract's control rate (DESIGN.md §3):
    ///   every mj_step:  write cached float64 ctrl   (the plug-in's MjActuator.OnSyncState overwrites mjData.ctrl
    ///                                                with its float32 `Control` after each step — so we re-assert)
    ///   every Nth step: read state → advance phase → build obs → infer → cache ctrl → apply disturbances
    /// Hooked on MjScene.preUpdateEvent, which fires immediately before each mj_step in MjScene.FixedUpdate.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public unsafe class PolicyRunner : MonoBehaviour
    {
        [Header("Contract & brain")]
        public TextAsset contractJson;
        public ModelAsset brain;
        public TextAsset brainSidecar;
        [Tooltip("Hold the default pose (ctrl = default) instead of running the brain.")]
        public bool holdDefaultPose;

        [Header("Athlete")]
        public string athletePrefix = "";
        public Vector3 command;
        [Tooltip("Race steering: overwrite command.z every control tick with the contract's lane-keeping yaw rate " +
                 "(heading along +x, back to this lane's centre line). Off for parity runs (fixed commands).")]
        public bool laneKeeping;
        /// <summary>Event steering (runtime only): called every control tick before the observation is built, with the
        /// root position relative to the lane origin, the heading and the current command; returns the new command.
        /// Same tick as the Python event loops, so event commands stay tick-exact. Runs after laneKeeping.</summary>
        public Func<double, double, double, Vector3, Vector3> steer;

        [Header("Stance skills (contract v4 brains only)")]
        [Tooltip("Stance-skill command (body units) appended to the observation of a contract v4 brain; ignored by v3 brains.")]
        public SkillCommand skill;

        [Header("All-fours events (30m All Fours)")]
        [Tooltip("Reset face down on the lane line, head towards +x (the finish), pelvis at proneHeight — " +
                 "training/poolympic/events/all_fours.py prone_start.")]
        public bool startProne;
        public double proneHeight = 0.22;
        [Tooltip("Lane keeping on the crawl heading (Contract.CrawlSteerYawRate) instead of the standing heading.")]
        public bool crawlSteering;

        [Header("Lane (meet scenes; solo = no prefix, origin 0, cube slots 0..3)")]
        [Tooltip("MuJoCo world x/y of this lane's origin: root position = contract default + origin.")]
        public double laneOriginX, laneOriginY;
        [Tooltip("Pool cube slots this lane owns: a lane-local script's cube<i> is cube<cubeSlots[i]> in the scene.")]
        public int[] cubeSlots = { 0, 1, 2, 3 };

        [Header("Athlete traits (DESIGN §1: per-lane stats → odds; nominal = parity)")]
        [Tooltip("Scales this athlete's actuator force limits (training DR: effort limits x[0.85, 1.15]).")]
        public float strength = 1f;
        [Tooltip("Physics substeps (0-4) before a new ctrl reaches the actuators (mjlab XmlActuator delay).")]
        public int latencySubsteps;
        [Tooltip("Scales the training observation noise (contract.obs_noise); 0 = clean.")]
        public float obsNoise;
        public int noiseSeed;

        [Header("Disturbances")]
        public bool useStandardParityScript = true;
        public List<Disturbance> disturbances = new();

        [Header("Brain confidence (broadcast telemetry; tools/export_critic.py)")]
        [Tooltip("The brain's PPO critic (<brain>.critic.onnx). Optional and read-only: it sees the brain's observation " +
                 "but never touches ctrl, so parity is unaffected.")]
        public ModelAsset critic;
        [Tooltip("Critic rate: every N control ticks (5 = 10 Hz), staggered per lane.")]
        public int criticEvery = 5;

        [Header("Recording (parity G5)")]
        public string recordName = "";
        public int recordTicks = 250;

        public int ControlTick { get; private set; }
        public bool Initialized { get; private set; }
        public Contract Contract { get; private set; }
        public AthleteBinding Binding { get; private set; }
        public float[] LastObs => _obs;
        /// <summary>Contract version of the loaded brain (sidecar): 3 = 84 obs, 4 = + stance-skill block.</summary>
        public int BrainVersion { get; private set; } = 3;
        /// <summary>Latest critic value (NaN without a critic) and how many times it has been evaluated.</summary>
        public float CriticValue { get; private set; } = float.NaN;
        public int CriticEvaluations { get; private set; }

        PolicyBrain _brain;
        PolicyCritic _critic;
        int _criticStagger;
        Dictionary<string, int> _jointIndex;
        readonly Dictionary<int, List<Disturbance>> _byTick = new();
        readonly List<Disturbance> _pending = new();
        bool _resetRequested;
        float[] _obs, _ctrlF, _actionRaw, _lastAction, _skill;
        double[] _ctrl, _ctrlPrev, _baseForceRange;
        System.Random _noiseRng;
        double _phase;
        int _substep;
        StringBuilder _rec;
        int _recFrames;

        void Awake()
        {
            Contract = Contract.Parse(contractJson.text);
            if (!holdDefaultPose) ValidateBrain();
            var scene = MjScene.Instance;
            scene.postInitEvent += OnPostInit;
            scene.preUpdateEvent += OnPreStep;
            var list = useStandardParityScript ? Disturbance.StandardParityScript(Contract.SpeedScale) : disturbances;
            foreach (var local in list)
            {
                var d = local.InLane(athletePrefix, cubeSlots, laneOriginX, laneOriginY);
                if (!_byTick.TryGetValue(d.tick, out var l)) _byTick[d.tick] = l = new List<Disturbance>();
                l.Add(d);
            }
        }

        void ValidateBrain()
        {
            if (brain == null || brainSidecar == null) throw new InvalidOperationException("PolicyRunner: brain + sidecar required");
            var sc = JsonUtility.FromJson<BrainSidecar>(brainSidecar.text);
            if (sc.fingerprint_sha256 != Contract.fingerprint_sha256)
                throw new InvalidOperationException($"Brain '{sc.name}' was trained on model {sc.fingerprint_sha256[..12]}…, contract is {Contract.fingerprint_sha256[..12]}… — refusing to run.");
            BrainVersion = int.Parse(sc.contract_version, CultureInfo.InvariantCulture);
            bool ok = BrainVersion == Contract.contract_version || (BrainVersion == 4 && Contract.SupportsSkills);
            if (!ok) throw new InvalidOperationException($"Brain '{sc.name}' contract version {sc.contract_version} does not match contract {Contract.contract_version}" +
                                                        (Contract.SupportsSkills ? " (+ skill block v4)" : ""));
            if (int.Parse(sc.decimation) != Contract.decimation || Math.Abs(double.Parse(sc.timestep, CultureInfo.InvariantCulture) - Contract.timestep) > 0)
                throw new InvalidOperationException("Brain timestep/decimation mismatch");
        }

        void OnPostInit(object sender, MjStepArgs args)
        {
            var m = args.model;
            var d = args.data;
            if (m->opt.timestep != Contract.timestep)
            {
                Debug.LogError($"[PolicyRunner] model timestep {m->opt.timestep:R} != contract {Contract.timestep:R} (check TimeManager Fixed Timestep ticks) — overriding.");
                m->opt.timestep = Contract.timestep;
            }
            Binding = new AthleteBinding(m, Contract, athletePrefix);
            _jointIndex = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, (int)m->njnt);
            ResetToDefault(m, d);

            int n = Contract.num_actions;
            _obs = new float[Contract.ObsDimFor(BrainVersion)];
            _skill = BrainVersion >= 4 ? new float[ObservationBuilder.SkillDim] : null;
            _ctrlF = new float[n];
            _actionRaw = new float[n];
            _lastAction = new float[n];
            _ctrl = new double[n];
            _ctrlPrev = new double[n];
            Array.Copy(Binding.DefaultPos, _ctrl, n);
            Array.Copy(Binding.DefaultPos, _ctrlPrev, n);
            _baseForceRange = new double[2 * n];
            for (int i = 0; i < n; i++)
            {
                _baseForceRange[2 * i] = m->actuator_forcerange[2 * Binding.ActuatorIds[i]];
                _baseForceRange[2 * i + 1] = m->actuator_forcerange[2 * Binding.ActuatorIds[i] + 1];
            }
            ApplyTraits(m);
            if (!holdDefaultPose) _brain = new PolicyBrain(brain, _obs.Length);
            if (!holdDefaultPose && critic != null && _critic == null)
            {
                _critic = new PolicyCritic(critic);
                if (_critic.ObsDim != _obs.Length) { _critic.Dispose(); _critic = null; }   // e.g. a v4 brain with a v3 critic
                foreach (var ch in athletePrefix) _criticStagger += ch;
            }
            if (!string.IsNullOrEmpty(recordName)) BeginRecording(m);
            ControlTick = 0;
            _substep = 0;
            _phase = 0;
            Initialized = true;
        }

        /// <summary>
        /// This athlete only (other lanes keep running): the contract's default state by joint name with the root
        /// shifted to the lane origin, zero velocity and solver warm start on its own dofs, then mj_forward.
        /// Mirrors training/poolympic/meet.py::reset_lane. At scene init mjData is fresh from mj_makeData, so this
        /// equals the solo reference's mj_resetDataKeyframe. Cubes are not touched (see <see cref="RequestReset"/>).
        /// </summary>
        public void ResetToDefault(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            foreach (var jq in Contract.default_joint_qpos)
            {
                if (jq.joint.StartsWith("cube")) continue;
                if (!_jointIndex.TryGetValue(athletePrefix + jq.joint, out var j)) throw new KeyNotFoundException($"joint '{athletePrefix + jq.joint}'");
                int qa = m->jnt_qposadr[j];
                bool free = m->jnt_type[j] == 0;
                for (int i = 0; i < jq.qpos.Length; i++)
                    d->qpos[qa + i] = jq.qpos[i] + (free && i == 0 ? laneOriginX : free && i == 1 ? laneOriginY : 0.0);
            }
            if (startProne)
            {
                // = mju_euler2Quat((0, π/2, 0), "XYZ"): pitch +90°, face down, spine (and head) along +x
                int r = Binding.RootQposAdr;
                d->qpos[r] = laneOriginX; d->qpos[r + 1] = laneOriginY; d->qpos[r + 2] = proneHeight;
                d->qpos[r + 3] = Math.Cos(Math.PI / 4); d->qpos[r + 4] = 0.0; d->qpos[r + 5] = Math.Sin(Math.PI / 4); d->qpos[r + 6] = 0.0;
            }
            foreach (var j in Binding.OwnJoints)
            {
                int da = m->jnt_dofadr[j];
                for (int i = 0; i < AthleteBinding.DofDim(m->jnt_type[j]); i++) d->qvel[da + i] = d->qacc_warmstart[da + i] = 0;
            }
            MujocoLib.mj_forward(m, d);
        }

        /// <summary>Set this athlete's traits (takes effect immediately; strength rescales its actuator force limits).</summary>
        public void SetTraits(float strengthScale, int latency, float noise, int seed)
        {
            strength = strengthScale;
            latencySubsteps = Math.Clamp(latency, 0, Contract != null ? Contract.decimation : 4);
            obsNoise = noise;
            noiseSeed = seed;
            if (Initialized && MjScene.InstanceExists) ApplyTraits(MjScene.Instance.Model);
        }

        void ApplyTraits(MujocoLib.mjModel_* m)
        {
            for (int i = 0; i < Binding.ActuatorIds.Length; i++)
            {
                m->actuator_forcerange[2 * Binding.ActuatorIds[i]] = _baseForceRange[2 * i] * strength;
                m->actuator_forcerange[2 * Binding.ActuatorIds[i] + 1] = _baseForceRange[2 * i + 1] * strength;
            }
            _noiseRng = new System.Random(noiseSeed);
        }

        /// <summary>Queue a native-MuJoCo disturbance for the next control tick. World frame: "root" means this
        /// athlete's root; cube targets are scene pool slots (MjCubePool).</summary>
        public void Request(Disturbance d)
        {
            if (d.target == "root") { d = d.Clone(); d.target = athletePrefix + "root"; }
            lock (_pending) _pending.Add(d);
        }

        /// <summary>Reset this athlete to the contract default state and park its own cube slots, at the next
        /// control tick.</summary>
        public void RequestReset()
        {
            _resetRequested = true;
            foreach (var c in cubeSlots) Request(new Disturbance { kind = "park", target = $"cube{c}_free" });
        }

        public double PelvisHeight(MujocoLib.mjData_* d) => d->qpos[Binding.RootQposAdr + 2];

        void OnPreStep(object sender, MjStepArgs args)
        {
            if (!Initialized) return;
            var m = args.model;
            var d = args.data;
            if (_substep % Contract.decimation == 0 && _resetRequested)
            {
                _resetRequested = false;
                ResetToDefault(m, d);
                Array.Copy(Binding.DefaultPos, _ctrl, _ctrl.Length);
                Array.Copy(Binding.DefaultPos, _ctrlPrev, _ctrlPrev.Length);
                Array.Clear(_lastAction, 0, _lastAction.Length);
                _phase = 0;
            }
            if (_substep % Contract.decimation == 0) ControlStep(m, d);
            // latency trait: a new ctrl reaches the actuators `latencySubsteps` physics steps after its control tick
            var applied = _substep % Contract.decimation >= latencySubsteps ? _ctrl : _ctrlPrev;
            for (int i = 0; i < applied.Length; i++) d->ctrl[Binding.ActuatorIds[i]] = applied[i];
            if (_substep % Contract.decimation == 0)
            {
                if (_byTick.TryGetValue(ControlTick, out var list))
                    foreach (var dist in list) dist.Apply(m, d, _jointIndex);
                lock (_pending)
                {
                    foreach (var dist in _pending) dist.Apply(m, d, _jointIndex);
                    _pending.Clear();
                }
            }
            _substep++;
            if (_substep % Contract.decimation == 0) ControlTick++;
        }

        /// <summary>Event command (MATT units) → this body's command: speeds × √λ, yaw rate ÷ √λ (Contract.SpeedScale),
        /// forward speed capped at the body's own top command (contract vx_max: GRANDMA 2.8 m/s). Identity for MATT.
        /// Python mirror: training/poolympic/events/iron_pedestal.py body_command.</summary>
        public Vector3 BodyCommand(Vector3 c)
        {
            float k = (float)Contract.SpeedScale;
            if (k == 1f) return c;
            float vx = c.x * k;
            if (Contract.vx_max > 0.0) vx = Mathf.Min(vx, (float)Contract.vx_max);
            return new Vector3(vx, c.y * k, c.z / k);
        }

        void ControlStep(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            if (_rec != null && _recFrames < recordTicks) RecordStateHead(m, d);
            if (laneKeeping)
            {
                int r = Binding.RootQposAdr;
                command.z = crawlSteering
                    ? (float)Contract.CrawlSteerYawRate(d->qpos[r + 3], d->qpos[r + 4], d->qpos[r + 5], d->qpos[r + 6], d->qpos[r + 1] - laneOriginY)
                    : (float)Contract.SteerYawRate(d->qpos[r + 3], d->qpos[r + 4], d->qpos[r + 5], d->qpos[r + 6],
                                                   d->qpos[r + 1] - laneOriginY, command.x);
            }
            if (steer != null)
            {
                int r = Binding.RootQposAdr;
                double w = d->qpos[r + 3], qx = d->qpos[r + 4], qy = d->qpos[r + 5], qz = d->qpos[r + 6];
                double yaw = Math.Atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
                command = steer(d->qpos[r] - laneOriginX, d->qpos[r + 1] - laneOriginY, yaw, command);
            }
            var bodyCommand = BodyCommand(command);
            if (_skill != null) skill.ToArray(_skill);
            _phase = Contract.AdvancePhase(_phase, bodyCommand, _skill != null ? skill.marchHz : 0.0);
            ObservationBuilder.Build(Contract, Binding, d->qpos, d->qvel, bodyCommand, _phase, _lastAction, _obs, _skill);
            if (obsNoise > 0f && Contract.obs_noise != null)
                foreach (var t in Contract.obs_noise)
                    for (int i = 0; i < t.size; i++)
                        _obs[t.offset + i] += (float)((_noiseRng.NextDouble() * 2.0 - 1.0) * t.amplitude * obsNoise);
            Array.Copy(_ctrl, _ctrlPrev, _ctrl.Length);
            if (holdDefaultPose)
            {
                Array.Copy(Binding.DefaultPos, _ctrl, _ctrl.Length);
                Array.Clear(_actionRaw, 0, _actionRaw.Length);
            }
            else
            {
                _brain.Run(_obs, _ctrlF, _actionRaw);
                for (int i = 0; i < _ctrl.Length; i++) _ctrl[i] = _ctrlF[i];
            }
            Array.Copy(_actionRaw, _lastAction, _lastAction.Length);
            if (_critic != null && (ControlTick + _criticStagger) % Math.Max(1, criticEvery) == 0)
            {
                CriticValue = _critic.Run(_obs);
                CriticEvaluations++;
            }
            if (_rec != null && _recFrames < recordTicks) RecordTail();
        }

        // ---------------------------------------------------------------- recording (G5 closed-loop comparison)
        /// <summary>Editor: the repo's parity/ folder; player builds (Android parity APK): the app's persistent data.</summary>
        public static string RecordDir => Application.isEditor ? Path.Combine(Application.dataPath, "..", "parity") : Application.persistentDataPath;
        public string RecordPath => _recPath;
        public bool RecordingDone => !string.IsNullOrEmpty(recordName) && _recFrames >= recordTicks;   // flushed at recordTicks
        string _recPath;
        int[] _recQ, _recV; // recorded qpos / qvel addresses: own joints + own cube slots, in joint order

        void BeginRecording(MujocoLib.mjModel_* m)
        {
            _recPath = Path.GetFullPath(Path.Combine(RecordDir, $"unity_run_{recordName}.json"));
            var joints = new List<int>(Binding.OwnJoints);
            foreach (var c in cubeSlots)
                if (_jointIndex.TryGetValue($"cube{c}_free", out var cj)) joints.Add(cj);
            joints.Sort();
            var q = new List<int>();
            var v = new List<int>();
            string R(double x) => x.ToString("R", CultureInfo.InvariantCulture);
            _rec = new StringBuilder();
            _rec.Append("{\"meta\":{\"source\":\"unity\",\"record_name\":\"").Append(recordName)
                .Append("\",\"timestep\":").Append(R(m->opt.timestep))
                .Append(",\"decimation\":").Append(Contract.decimation)
                .Append(",\"prefix\":\"").Append(athletePrefix)
                .Append("\",\"origin\":[").Append(R(laneOriginX)).Append(',').Append(R(laneOriginY)).Append(",0]")
                .Append(",\"cube_slots\":[").Append(string.Join(",", cubeSlots))
                .Append("],\"joints\":[");
            for (int k = 0; k < joints.Count; k++)
            {
                int j = joints[k], t = m->jnt_type[j];
                if (k > 0) _rec.Append(',');
                _rec.Append("{\"name\":\"").Append(ModelFingerprint.Name(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, j))
                    .Append("\",\"type\":").Append(t).Append(",\"qposadr\":").Append(q.Count)
                    .Append(",\"dofadr\":").Append(v.Count).Append('}');
                for (int i = 0; i < AthleteBinding.QposDim(t); i++) q.Add(m->jnt_qposadr[j] + i);
                for (int i = 0; i < AthleteBinding.DofDim(t); i++) v.Add(m->jnt_dofadr[j] + i);
            }
            _recQ = q.ToArray();
            _recV = v.ToArray();
            _rec.Append("]},\"frames\":[");
            _recFrames = 0;
        }


        void RecordStateHead(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            if (_recFrames > 0) _rec.Append(',');
            _rec.Append("{\"tick\":").Append(ControlTick)
                .Append(",\"t\":").Append((ControlTick * Contract.timestep * Contract.decimation).ToString("R", CultureInfo.InvariantCulture));
            AppendGather("qpos", d->qpos, _recQ);
            AppendGather("qvel", d->qvel, _recV);
            var f = stackalloc double[Binding.ActuatorIds.Length];
            for (int i = 0; i < Binding.ActuatorIds.Length; i++) f[i] = d->actuator_force[Binding.ActuatorIds[i]];
            AppendArray("actuator_force_prev", f, Binding.ActuatorIds.Length);
        }

        void RecordTail()
        {
            _rec.Append(",\"phase\":").Append(_phase.ToString("R", CultureInfo.InvariantCulture));
            AppendFloats("obs", _obs);
            AppendFloats("action_raw", _actionRaw);
            var c = stackalloc double[_ctrl.Length];
            for (int i = 0; i < _ctrl.Length; i++) c[i] = _ctrl[i];
            AppendArray("ctrl", c, _ctrl.Length);
            _rec.Append('}');
            _recFrames++;
            if (_recFrames == recordTicks) FlushRecording();
        }

        void AppendArray(string key, double* p, int n)
        {
            _rec.Append(",\"").Append(key).Append("\":[");
            for (int i = 0; i < n; i++)
            {
                if (i > 0) _rec.Append(',');
                _rec.Append(p[i].ToString("R", CultureInfo.InvariantCulture));
            }
            _rec.Append(']');
        }

        void AppendGather(string key, double* p, int[] idx)
        {
            _rec.Append(",\"").Append(key).Append("\":[");
            for (int i = 0; i < idx.Length; i++)
            {
                if (i > 0) _rec.Append(',');
                _rec.Append(p[idx[i]].ToString("R", CultureInfo.InvariantCulture));
            }
            _rec.Append(']');
        }

        void AppendFloats(string key, float[] a)
        {
            _rec.Append(",\"").Append(key).Append("\":[");
            for (int i = 0; i < a.Length; i++)
            {
                if (i > 0) _rec.Append(',');
                _rec.Append(((double)a[i]).ToString("R", CultureInfo.InvariantCulture));
            }
            _rec.Append(']');
        }

        void FlushRecording()
        {
            if (_rec == null) return;
            _rec.Append("]}");
            Directory.CreateDirectory(Path.GetDirectoryName(_recPath));
            File.WriteAllText(_recPath, _rec.ToString());
            Debug.Log($"[PolicyRunner] recorded {_recFrames} control ticks → {_recPath}");
            _rec = null;
        }

        void OnDestroy()
        {
            FlushRecording();
            _brain?.Dispose();
            _critic?.Dispose();
            if (MjScene.InstanceExists)
            {
                MjScene.Instance.postInitEvent -= OnPostInit;
                MjScene.Instance.preUpdateEvent -= OnPreStep;
            }
        }
    }
}
