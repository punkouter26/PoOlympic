using System;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// 84-dim policy observation, built only from mjData qpos/qvel (never Unity Transforms). Line-for-line port of
    /// training/poolympic/contract.py::build_obs — computed in double, cast to float at the end. Gate G2.
    /// Contract v4 brains: the 11-value stance-skill block (SkillCommand) is appended → 95.
    /// </summary>
    public static unsafe class ObservationBuilder
    {
        public const int ObsDim = 84;
        public const int SkillDim = 11;

        public static void Build(Contract c, AthleteBinding a, double* qpos, double* qvel, Vector3 command,
                                 double phase, float[] lastAction, float[] obs, float[] skill = null)
        {
            int r = a.RootQposAdr, dv = a.RootDofAdr;
            double* quat = qpos + r + 3;
            var R = stackalloc double[9];
            MujocoLib.mju_quat2Mat(R, quat);

            double vx = qvel[dv], vy = qvel[dv + 1], vz = qvel[dv + 2]; // free joint: world-frame linear velocity
            double w = quat[0], x = quat[1], y = quat[2], z = quat[3];
            double yaw = Math.Atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
            double cy = Math.Cos(yaw), sy = Math.Sin(yaw);

            int o = 0;
            obs[o++] = (float)(cy * vx + sy * vy);          // base_lin_vel_heading
            obs[o++] = (float)(-sy * vx + cy * vy);
            obs[o++] = (float)vz;
            obs[o++] = (float)qvel[dv + 3];                 // base_ang_vel_local (free joint: body frame)
            obs[o++] = (float)qvel[dv + 4];
            obs[o++] = (float)qvel[dv + 5];
            obs[o++] = (float)(-R[6]);                      // projected_gravity = R^T (0,0,-1) = -(R row 2)
            obs[o++] = (float)(-R[7]);
            obs[o++] = (float)(-R[8]);
            obs[o++] = (float)qpos[r + 2];                  // base_height
            obs[o++] = command.x;                           // command
            obs[o++] = command.y;
            obs[o++] = command.z;
            obs[o++] = (float)Math.Sin(2.0 * Math.PI * phase); // gait_phase_sincos
            obs[o++] = (float)Math.Cos(2.0 * Math.PI * phase);
            int n = a.ActuatorIds.Length;
            for (int i = 0; i < n; i++) obs[o++] = (float)(qpos[a.JointQposAdr[i]] - a.DefaultPos[i]);        // joint_pos_rel
            for (int i = 0; i < n; i++) obs[o++] = (float)(qvel[a.JointDofAdr[i]] * c.joint_vel_scale);     // joint_vel_scaled
            for (int i = 0; i < n; i++) obs[o++] = lastAction[i];                                            // last_action
            if (o != ObsDim) throw new InvalidOperationException($"obs size {o} != {ObsDim}");
            if (skill != null)
            {
                if (skill.Length != SkillDim) throw new ArgumentException($"skill block {skill.Length} != {SkillDim}");
                for (int i = 0; i < SkillDim; i++) obs[o++] = skill[i];                                       // skill (v4)
            }
            if (o != obs.Length) throw new InvalidOperationException($"obs buffer {obs.Length} != built {o}");
        }
    }
}
