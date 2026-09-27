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

        [Header("Lane (meet scenes; solo = no prefix, origin 0, cube slots 0..3)")]
        [Tooltip("MuJoCo world x/y of this lane's origin: root position = contract default + origin.")]
        public double laneOriginX, laneOriginY;
        [Tooltip("Pool cube slots this lane owns: a lane-local script's cube<i> is cube<cubeSlots[i]> in the scene.")]
        public int[] cubeSlots = { 0, 1, 2, 3 };

        [Header("Disturbances")]
        public bool useStandardParityScript = true;
        public List<Disturbance> disturbances = new();

        [Header("Recording (parity G5)")]
        public string recordName = "";
        public int recordTicks = 250;

        public int ControlTick { get; private set; }
        public bool Initialized { get; private set; }
        public Contract Contract { get; private set; }
        public AthleteBinding Binding { get; private set; }
        public float[] LastObs => _obs;

        PolicyBrain _brain;
        Dictionary<string, int> _jointIndex;
        readonly Dictionary<int, List<Disturbance>> _byTick = new();
        readonly List<Disturbance> _pending = new();
        bool _resetRequested;
        float[] _obs, _ctrlF, _actionRaw, _lastAction;
        double[] _ctrl;
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
            var list = useStandardParityScript ? Disturbance.StandardParityScript() : disturbances;
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
            if (sc.contract_version != Contract.contract_version.ToString())
                throw new InvalidOperationException("Brain contract version mismatch");
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
            _obs = new float[Contract.obs_dim];
            _ctrlF = new float[n];
            _actionRaw = new float[n];
            _lastAction = new float[n];
            _ctrl = new double[n];
            Array.Copy(Binding.DefaultPos, _ctrl, n);
            if (!holdDefaultPose) _brain = new PolicyBrain(brain, Contract.obs_dim);
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
            foreach (var j in Binding.OwnJoints)
            {
                int da = m->jnt_dofadr[j];
                for (int i = 0; i < AthleteBinding.DofDim(m->jnt_type[j]); i++) d->qvel[da + i] = d->qacc_warmstart[da + i] = 0;
            }
            MujocoLib.mj_forward(m, d);
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
                Array.Clear(_lastAction, 0, _lastAction.Length);
                _phase = 0;
            }
            if (_substep % Contract.decimation == 0) ControlStep(m, d);
            for (int i = 0; i < _ctrl.Length; i++) d->ctrl[Binding.ActuatorIds[i]] = _ctrl[i];
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

        void ControlStep(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            if (_rec != null && _recFrames < recordTicks) RecordStateHead(m, d);
            if (laneKeeping)
            {
                int r = Binding.RootQposAdr;
                command.z = (float)Contract.SteerYawRate(d->qpos[r + 3], d->qpos[r + 4], d->qpos[r + 5], d->qpos[r + 6],
                                                         d->qpos[r + 1] - laneOriginY);
            }
            _phase = Contract.AdvancePhase(_phase, command);
            ObservationBuilder.Build(Contract, Binding, d->qpos, d->qvel, command, _phase, _lastAction, _obs);
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
            if (_rec != null && _recFrames < recordTicks) RecordTail();
        }

        // ---------------------------------------------------------------- recording (G5 closed-loop comparison)
        string _recPath;
        int[] _recQ, _recV; // recorded qpos / qvel addresses: own joints + own cube slots, in joint order

        void BeginRecording(MujocoLib.mjModel_* m)
        {
            _recPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "parity", $"unity_run_{recordName}.json"));
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
            if (MjScene.InstanceExists)
            {
                MjScene.Instance.postInitEvent -= OnPostInit;
                MjScene.Instance.preUpdateEvent -= OnPreStep;
            }
        }
    }
}
