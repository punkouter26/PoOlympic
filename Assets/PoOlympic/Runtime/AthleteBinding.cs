using System;
using System.Collections.Generic;
using Mujoco;

namespace PoOlympic
{
    /// <summary>
    /// Index bindings of one athlete inside the compiled mjModel, resolved by (suffix-stripped) name so the
    /// layout never has to match Python's. Mirrors contract.Athlete.bind.
    /// </summary>
    public sealed unsafe class AthleteBinding
    {
        public readonly int[] ActuatorIds;
        public readonly int[] JointQposAdr;
        public readonly int[] JointDofAdr;
        public readonly double[] DefaultPos;
        public readonly int RootQposAdr;
        public readonly int RootDofAdr;
        /// <summary>Every joint of this athlete (root, actuated and passive): joints whose body hangs under its pelvis.</summary>
        public readonly int[] OwnJoints;
        public readonly string Prefix;

        public AthleteBinding(MujocoLib.mjModel_* m, Contract c, string prefix = "")
        {
            Prefix = prefix;
            var actuators = NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_ACTUATOR, (int)m->nu);
            var joints = NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, (int)m->njnt);
            int n = c.actuators.Length;
            ActuatorIds = new int[n];
            JointQposAdr = new int[n];
            JointDofAdr = new int[n];
            DefaultPos = new double[n];
            for (int i = 0; i < n; i++)
            {
                var a = c.actuators[i];
                if (!actuators.TryGetValue(prefix + a.name, out var aid)) throw new KeyNotFoundException($"actuator '{prefix + a.name}' not in model");
                if (!joints.TryGetValue(prefix + a.joint, out var jid)) throw new KeyNotFoundException($"joint '{prefix + a.joint}' not in model");
                if (m->actuator_trnid[2 * aid] != jid) throw new InvalidOperationException($"actuator {a.name} does not drive joint {a.joint}");
                ActuatorIds[i] = aid;
                JointQposAdr[i] = m->jnt_qposadr[jid];
                JointDofAdr[i] = m->jnt_dofadr[jid];
                DefaultPos[i] = a.default_rad;
            }
            if (!joints.TryGetValue(prefix + c.root_joint, out var root)) throw new KeyNotFoundException($"root joint '{prefix + c.root_joint}' not in model");
            RootQposAdr = m->jnt_qposadr[root];
            RootDofAdr = m->jnt_dofadr[root];
            int pelvis = m->jnt_bodyid[root];
            var own = new List<int>();
            for (int j = 0; j < (int)m->njnt; j++)
                if (m->body_rootid[m->jnt_bodyid[j]] == pelvis) own.Add(j);
            OwnJoints = own.ToArray();
        }

        public static int QposDim(int jntType) => jntType == 0 ? 7 : jntType == 1 ? 4 : 1;
        public static int DofDim(int jntType) => jntType == 0 ? 6 : jntType == 1 ? 3 : 1;

        public static Dictionary<string, int> NameIndex(MujocoLib.mjModel_* m, int objType, int count)
        {
            var d = new Dictionary<string, int>(count);
            for (int i = 0; i < count; i++) d[ModelFingerprint.Name(m, objType, i)] = i;
            return d;
        }
    }
}
