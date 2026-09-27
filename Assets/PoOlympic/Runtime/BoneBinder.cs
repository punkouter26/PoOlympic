using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Drives a skinned visual rig from MuJoCo bodies (DESIGN.md §2/§4). Only the pivot bone of each physics body is
    /// driven; bones merged into a body (Spine1, Head, fingers, …) keep their bind-local transforms and follow rigidly.
    /// Offsets are captured once at qpos = 0 (the bind/T-pose), where every MjBody frame is world-aligned:
    ///   bone.world = body.world · offset
    /// </summary>
    [DefaultExecutionOrder(100)]
    public class BoneBinder : MonoBehaviour
    {
        [Serializable]
        public class Binding
        {
            public Transform bone;
            public MjBody body;
            public Vector3 offsetPos;
            public Quaternion offsetRot = Quaternion.identity;
        }

        [Tooltip("Parent-before-child order.")]
        public List<Binding> bindings = new();

        /// <summary>Pivot bone (MATT/Mixamo naming) → physics body name. Mirrors build_mjcf.body_specs().</summary>
        public static readonly Dictionary<string, string> PivotBones = new()
        {
            { "Hips", "pelvis" }, { "Spine", "torso" }, { "Spine2", "chest" }, { "Neck", "head" },
            { "LeftArm", "upper_arm_l" }, { "LeftForeArm", "forearm_l" }, { "RightArm", "upper_arm_r" }, { "RightForeArm", "forearm_r" },
            { "LeftUpLeg", "thigh_l" }, { "LeftLeg", "shin_l" }, { "LeftFoot", "foot_l" }, { "LeftToeBase", "toe_l" },
            { "RightUpLeg", "thigh_r" }, { "RightLeg", "shin_r" }, { "RightFoot", "foot_r" }, { "RightToeBase", "toe_r" },
        };

        /// <summary>Capture offsets. Must be called while the MjBody transforms show qpos = 0 (Edit mode after import).</summary>
        public void Capture(Transform physicsRoot)
        {
            bindings.Clear();
            var bodies = new Dictionary<string, MjBody>();
            foreach (var b in physicsRoot.GetComponentsInChildren<MjBody>(true)) bodies[b.name] = b;
            foreach (var bone in GetComponentsInChildren<Transform>(true)) // depth-first: parents before children
            {
                if (!PivotBones.TryGetValue(bone.name, out var bodyName)) continue;
                if (!bodies.TryGetValue(bodyName, out var body)) throw new KeyNotFoundException($"MjBody '{bodyName}' for bone '{bone.name}'");
                var inv = Quaternion.Inverse(body.transform.rotation);
                bindings.Add(new Binding
                {
                    bone = bone, body = body,
                    offsetPos = inv * (bone.position - body.transform.position),
                    offsetRot = inv * bone.rotation,
                });
            }
        }

        /// <summary>Largest bind-pose position error (m) and rotation of the body frame (deg) — B5 acceptance.</summary>
        public (float posErr, float rotErr) BindError()
        {
            float p = 0, r = 0;
            foreach (var b in bindings)
            {
                p = Mathf.Max(p, (b.bone.position - b.body.transform.position).magnitude);
                r = Mathf.Max(r, Quaternion.Angle(b.body.transform.rotation, Quaternion.identity));
            }
            return (p, r);
        }

        void LateUpdate()
        {
            foreach (var b in bindings)
            {
                var t = b.body.transform;
                b.bone.SetPositionAndRotation(t.position + t.rotation * b.offsetPos, t.rotation * b.offsetRot);
            }
        }
    }
}
