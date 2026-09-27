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
            foreach (var d in list)
            {
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

        /// <summary>mj_resetData, then the contract's default state by joint name, then mj_forward.</summary>
        public void ResetToDefault(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            MujocoLib.mj_resetData(m, d);
            foreach (var jq in Contract.default_joint_qpos)
            {
                var name = jq.joint == Contract.root_joint || !jq.joint.StartsWith("cube") ? athletePrefix + jq.joint : jq.joint;
                if (!_jointIndex.TryGetValue(name, out var j)) continue;
                int qa = m->jnt_qposadr[j];
                for (int i = 0; i < jq.qpos.Length; i++) d->qpos[qa + i] = jq.qpos[i];
            }
            MujocoLib.mj_forward(m, d);
        }

        void OnPreStep(object sender, MjStepArgs args)
        {
            if (!Initialized) return;
            var m = args.model;
            var d = args.data;
            if (_substep % Contract.decimation == 0) ControlStep(m, d);
            for (int i = 0; i < _ctrl.Length; i++) d->ctrl[Binding.ActuatorIds[i]] = _ctrl[i];
            if (_substep % Contract.decimation == 0 && _byTick.TryGetValue(ControlTick, out var list))
                foreach (var dist in list) dist.Apply(m, d, _jointIndex, athletePrefix);
            _substep++;
            if (_substep % Contract.decimation == 0) ControlTick++;
        }

        void ControlStep(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            if (_rec != null && _recFrames < recordTicks) RecordStateHead(m, d);
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
        int _nq, _nv;

        void BeginRecording(MujocoLib.mjModel_* m)
        {
            _nq = (int)m->nq;
            _nv = (int)m->nv;
            _recPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "parity", $"unity_run_{recordName}.json"));
            _rec = new StringBuilder();
            _rec.Append("{\"meta\":{\"source\":\"unity\",\"record_name\":\"").Append(recordName)
                .Append("\",\"timestep\":").Append(m->opt.timestep.ToString("R", CultureInfo.InvariantCulture))
                .Append(",\"decimation\":").Append(Contract.decimation)
                .Append(",\"joints\":[");
            for (int j = 0; j < (int)m->njnt; j++)
            {
                if (j > 0) _rec.Append(',');
                _rec.Append("{\"name\":\"").Append(ModelFingerprint.Name(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, j))
                    .Append("\",\"type\":").Append(m->jnt_type[j]).Append(",\"qposadr\":").Append(m->jnt_qposadr[j])
                    .Append(",\"dofadr\":").Append(m->jnt_dofadr[j]).Append('}');
            }
            _rec.Append("]},\"frames\":[");
            _recFrames = 0;
        }


        void RecordStateHead(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            if (_recFrames > 0) _rec.Append(',');
            _rec.Append("{\"tick\":").Append(ControlTick)
                .Append(",\"t\":").Append((ControlTick * Contract.timestep * Contract.decimation).ToString("R", CultureInfo.InvariantCulture));
            AppendArray("qpos", d->qpos, _nq);
            AppendArray("qvel", d->qvel, _nv);
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
