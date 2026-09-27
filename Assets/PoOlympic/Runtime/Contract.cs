using System;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// C# view of parity/contract.json (written by training/poolympic/contract.py). The Python module is the
    /// source of truth; this class only reads it. Field names match the JSON keys.
    /// </summary>
    [Serializable]
    public class Contract
    {
        [Serializable]
        public class Actuator
        {
            public string name;
            public string joint;
            public double default_rad;
            public double range_lo_rad;
            public double range_hi_rad;
        }

        [Serializable]
        public class JointQpos
        {
            public string joint;
            public double[] qpos;
        }

        public int contract_version;
        public string mujoco_version;
        public double timestep;
        public int decimation;
        public double action_scale;
        public double joint_vel_scale;
        public double gait_hz;
        public double phase_cmd_threshold;
        public int obs_dim;
        public int num_actions;
        public string root_joint;
        public Actuator[] actuators;
        public double[] default_qpos_scene;
        public JointQpos[] default_joint_qpos;
        public string fingerprint_sha256;

        public static Contract Parse(string json)
        {
            var c = JsonUtility.FromJson<Contract>(json);
            if (c.actuators == null || c.actuators.Length != c.num_actions)
                throw new InvalidOperationException("contract.json: actuator list does not match num_actions");
            if (c.obs_dim != ObservationBuilder.ObsDim)
                throw new InvalidOperationException($"contract.json obs_dim {c.obs_dim} != C# ObservationBuilder {ObservationBuilder.ObsDim}");
            return c;
        }

        /// <summary>Phase clock update, called once per control tick before building the observation.</summary>
        public double AdvancePhase(double phase, Vector3 command)
        {
            double n = Math.Sqrt((double)command.x * command.x + (double)command.y * command.y + (double)command.z * command.z);
            if (n < phase_cmd_threshold) return 0.0;
            double p = phase + gait_hz * decimation * timestep;
            return p - Math.Floor(p);
        }
    }

    /// <summary>Sidecar written next to every exported ONNX (training/poolympic/policy_export.py).</summary>
    [Serializable]
    public class BrainSidecar
    {
        public string name;
        public string contract_version;
        public string fingerprint_sha256;
        public string mujoco_version;
        public string timestep;
        public string decimation;
        public string onnx_sha256;
    }
}
