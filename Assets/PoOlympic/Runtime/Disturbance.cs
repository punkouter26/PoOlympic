using System;
using System.Collections.Generic;
using Mujoco;

namespace PoOlympic
{
    /// <summary>
    /// Scheduled physical disturbance applied through native MuJoCo state only (mirrors
    /// training/poolympic/reference.py::Disturbance). Applied at a control-tick boundary, after ctrl is written and
    /// before the tick's mj_steps. `target` is a joint name of the compiled scene; scripts written in lane-local
    /// (solo-scene) terms are mapped into a meet lane with <see cref="InLane"/>.
    /// </summary>
    [Serializable]
    public class Disturbance
    {
        public int tick;
        public string kind;   // "shove": Δqvel on the free joint's linear dofs | "kick": Δqvel on a 1-dof joint | "cube": teleport a pooled cube (qpos7 + qvel6)
                              // "park": return a pooled cube to its parking slot (model qpos0, at rest)
        public string target; // joint name, e.g. "root" or "cube0_free"
        public double[] dqvel;
        public double[] qpos;
        public double[] qvel;

        public Disturbance Clone() => new()
        {
            tick = tick, kind = kind, target = target,
            dqvel = (double[])dqvel?.Clone(), qpos = (double[])qpos?.Clone(), qvel = (double[])qvel?.Clone(),
        };

        /// <summary>
        /// Lane-local → meet (training/poolympic/meet.py::Lane.disturbance): root → prefix + root,
        /// cube&lt;i&gt;_free → cube&lt;cubeSlots[i]&gt;_free, teleport position += lane origin.
        /// </summary>
        public Disturbance InLane(string prefix, int[] cubeSlots, double originX, double originY)
        {
            var o = Clone();
            if (target.StartsWith("cube"))
            {
                int i = int.Parse(target.Substring(4, target.Length - 4 - "_free".Length));
                if (i >= cubeSlots.Length) throw new ArgumentException($"lane owns no cube slot {i}");
                o.target = $"cube{cubeSlots[i]}_free";
                if (kind == "cube") { o.qpos[0] += originX; o.qpos[1] += originY; }
            }
            else o.target = prefix + target;
            return o;
        }

        public unsafe void Apply(MujocoLib.mjModel_* m, MujocoLib.mjData_* d, Dictionary<string, int> jointIndex)
        {
            if (!jointIndex.TryGetValue(target, out var j)) throw new KeyNotFoundException($"disturbance target joint '{target}'");
            int qa = m->jnt_qposadr[j], da = m->jnt_dofadr[j];
            switch (kind)
            {
                case "shove":
                    for (int i = 0; i < 3; i++) d->qvel[da + i] += dqvel[i];
                    break;
                case "kick":   // Δqvel on a 1-dof joint (slide / hinge), e.g. a shaker platform axis
                    d->qvel[da] += dqvel[0];
                    break;
                case "cube":
                    for (int i = 0; i < 7; i++) d->qpos[qa + i] = qpos[i];
                    for (int i = 0; i < 6; i++) d->qvel[da + i] = qvel[i];
                    break;
                case "park":
                    for (int i = 0; i < 7; i++) d->qpos[qa + i] = m->qpos0[qa + i];
                    for (int i = 0; i < 6; i++) d->qvel[da + i] = 0;
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
