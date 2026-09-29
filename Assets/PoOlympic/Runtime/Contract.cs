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
        public class Steering
        {
            public double heading_gain;
            public double lane_gain;
            public double wz_limit;
        }

        [Serializable]
        public class ObsNoiseTerm
        {
            public string term;
            public int offset;
            public int size;
            public double amplitude;
        }

        [Serializable]
        public class TraitRanges
        {
            public double[] strength;
            public double[] latency_substeps;
            public double[] obs_noise;
        }

        [Serializable]
        public class JointQpos
        {
            public string joint;
            public double[] qpos;
        }

        /// <summary>Contract v4 stance-skill command block (docs/CONTRACT_V4_STANCE_PROPOSAL.md): appended after the v3
        /// obs for brains whose sidecar says contract_version 4. JsonUtility leaves version 0 when the key is absent.</summary>
        [Serializable]
        public class SkillBlock
        {
            public int version;
            public int offset;
            public int size;
            public int obs_dim;
        }

        public string body;                   // athlete body (Phase Z); absent = MATT
        public int contract_version;
        public string mujoco_version;
        public double timestep;
        public int decimation;
        public double action_scale;
        public double joint_vel_scale;
        public double gait_hz_base;
        public double gait_hz_per_mps;
        public double gait_hz_yaw_weight;
        public double phase_cmd_threshold;
        public Steering steering;
        public ObsNoiseTerm[] obs_noise;
        public TraitRanges trait_ranges;
        public int obs_dim;
        public int num_actions;
        public string root_joint;
        public Actuator[] actuators;
        public double[] default_qpos_scene;
        public JointQpos[] default_joint_qpos;
        public string fingerprint_sha256;
        public SkillBlock skill_block;

        public bool SupportsSkills => skill_block != null && skill_block.version > contract_version && skill_block.size == ObservationBuilder.SkillDim
                                      && skill_block.offset == obs_dim;

        /// <summary>Policy input size for a brain of `brainVersion`: the v3 obs, or v3 + skill block for v4 brains.</summary>
        public int ObsDimFor(int brainVersion) => brainVersion >= 4 && SupportsSkills ? skill_block.obs_dim : obs_dim;

        public string BodyName => string.IsNullOrEmpty(body) ? "matt" : body;

        /// <summary>Fall rule pelvis height: MATT 0.55 m; other bodies the same fraction of their standing pelvis height
        /// (training/poolympic/events/iron_pedestal.py body_fall_z).</summary>
        public double FallPelvisZ
        {
            get
            {
                if (BodyName == "matt") return 0.55;
                foreach (var j in default_joint_qpos)
                    if (j.joint == root_joint) return 0.55 * j.qpos[2] / MattRootZ;
                throw new InvalidOperationException("contract: root joint missing from default_joint_qpos");
            }
        }

        public const double MattRootZ = 0.9549291;   // scene_matt.xml keyframe "default"

        /// <summary>Froude speed scale √λ of this body vs MATT (λ = height ratio): event commands are written in MATT
        /// units and scaled per body (speeds × √λ, yaw rates ÷ √λ), matching the body's training envelope. Recovered from
        /// the gait clock: training/poolympic/contract.py GAIT_HZ_BASE = 0.8 / √λ.</summary>
        public double SpeedScale => BodyName == "matt" ? 1.0 : MattGaitHzBase / gait_hz_base;

        public const double MattGaitHzBase = 0.8;

        public static Contract Parse(string json)
        {
            var c = JsonUtility.FromJson<Contract>(json);
            if (c.actuators == null || c.actuators.Length != c.num_actions)
                throw new InvalidOperationException("contract.json: actuator list does not match num_actions");
            if (c.obs_dim != ObservationBuilder.ObsDim)
                throw new InvalidOperationException($"contract.json obs_dim {c.obs_dim} != C# ObservationBuilder {ObservationBuilder.ObsDim}");
            return c;
        }

        /// <summary>Phase clock update, called once per control tick before building the observation
        /// (contract.py::advance_phase / gait_hz). v4: a march cadence &gt; 0 (Hz) drives the clock at that rate whatever
        /// the command (marching in place).</summary>
        public double AdvancePhase(double phase, Vector3 command, double cadence = 0.0)
        {
            if (cadence > 0.0)
            {
                double q = phase + cadence * decimation * timestep;
                return q - Math.Floor(q);
            }
            double n = Math.Sqrt((double)command.x * command.x + (double)command.y * command.y + (double)command.z * command.z);
            if (n < phase_cmd_threshold) return 0.0;
            double p = phase + GaitHz(command) * decimation * timestep;
            return p - Math.Floor(p);
        }

        /// <summary>Lane-keeping yaw-rate command (contract.py::steer_yaw_rate): steer the pelvis heading (yaw of its
        /// x-axis) towards atan(-lane_gain · lane offset · dir) along world +x; dir = -1 when running backwards.</summary>
        public double SteerYawRate(double w, double x, double y, double z, double laneOffsetY, double vxCommand)
        {
            double yaw = Math.Atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
            double dir = vxCommand < 0 ? -1.0 : 1.0;
            double target = Math.Atan(-steering.lane_gain * laneOffsetY * dir);
            double err = target - yaw + Math.PI;
            err = err - 2 * Math.PI * Math.Floor(err / (2 * Math.PI)) - Math.PI;
            return Math.Clamp(steering.heading_gain * err, -steering.wz_limit, steering.wz_limit);
        }

        /// <summary>Lane keeping on all fours (event 8, 30m All Fours) — training/poolympic/events/all_fours.py crawl_steer:
        /// the contract law on the crawl heading = horizontal projection of the pelvis x + z axes (the x axis alone points at
        /// the ground on all fours and flips towards the feet when the hips are above the shoulders). Forward only.</summary>
        public double CrawlSteerYawRate(double w, double x, double y, double z, double laneOffsetY)
        {
            double hx = (1.0 - 2.0 * (y * y + z * z)) + 2.0 * (x * z + w * y);
            double hy = 2.0 * (x * y + w * z) + 2.0 * (y * z - w * x);
            double yaw = Math.Atan2(hy, hx);
            double target = Math.Atan(-steering.lane_gain * laneOffsetY);
            double err = target - yaw + Math.PI;
            err = err - 2 * Math.PI * Math.Floor(err / (2 * Math.PI)) - Math.PI;
            return Math.Clamp(steering.heading_gain * err, -steering.wz_limit, steering.wz_limit);
        }

        public double GaitHz(Vector3 command)
        {
            double speed = Math.Sqrt((double)command.x * command.x + (double)command.y * command.y) + gait_hz_yaw_weight * Math.Abs((double)command.z);
            return gait_hz_base + gait_hz_per_mps * speed;
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
