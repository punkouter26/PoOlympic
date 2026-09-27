using System;
using System.Collections.Generic;
using Mujoco;

namespace PoOlympic
{
    /// <summary>
    /// Scheduled physical disturbance applied through native MuJoCo state only (mirrors
    /// training/poolympic/reference.py::Disturbance). Applied at a control-tick boundary, after ctrl is written and
    /// before the tick's mj_steps.
    /// </summary>
    [Serializable]
    public class Disturbance
    {
        public int tick;
        public string kind;   // "shove": Δqvel on the free joint's linear dofs | "cube": teleport a pooled cube (qpos7 + qvel6)
        public string target; // joint name, e.g. "root" or "cube0_free"
        public double[] dqvel;
        public double[] qpos;
        public double[] qvel;

        public unsafe void Apply(MujocoLib.mjModel_* m, MujocoLib.mjData_* d, Dictionary<string, int> jointIndex, string prefix)
        {
            var name = target == "root" ? prefix + target : target;
            if (!jointIndex.TryGetValue(name, out var j)) throw new KeyNotFoundException($"disturbance target joint '{name}'");
            int qa = m->jnt_qposadr[j], da = m->jnt_dofadr[j];
            switch (kind)
            {
                case "shove":
                    for (int i = 0; i < 3; i++) d->qvel[da + i] += dqvel[i];
                    break;
                case "cube":
                    for (int i = 0; i < 7; i++) d->qpos[qa + i] = qpos[i];
                    for (int i = 0; i < 6; i++) d->qvel[da + i] = qvel[i];
                    break;
                default:
                    throw new ArgumentException($"unknown disturbance kind '{kind}'");
            }
        }

        /// <summary>The standard parity script — identical to reference.default_disturbances().</summary>
        public static List<Disturbance> StandardParityScript() => new()
        {
            new Disturbance { tick = 50, kind = "shove", target = "root", dqvel = new[] { 0.0, 0.5, 0.0 } },
            new Disturbance { tick = 100, kind = "cube", target = "cube0_free",
                              qpos = new[] { 0.0, 0.2, 3.0, 1.0, 0.0, 0.0, 0.0 }, qvel = new double[6] },
        };
    }
}
