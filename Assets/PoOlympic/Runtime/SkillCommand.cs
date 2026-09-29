using System;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Contract v4 stance-skill command (training/poolympic/contract.py SkillCommand, docs/CONTRACT_V4_STANCE_PROPOSAL.md)
    /// in the athlete body's own units. ToArray() is the 11-value obs block appended for v4 brains; all zero = no skill.
    /// </summary>
    [Serializable]
    public struct SkillCommand
    {
        [Tooltip("Target pelvis height relative to standing (m, ≤ 0) — event 3 Deep Squat.")]
        public float pelvisHeight;
        [Tooltip("Lift this foot and stand on the other leg — event 6 Flamingo.")]
        public Foot liftFoot;
        [Tooltip("March cadence (Hz); > 0 drives the gait clock at zero velocity — event 7 Cadence March.")]
        public float marchHz;
        [Tooltip("March knee lift (m) — event 7.")]
        public float kneeLift;
        [Tooltip("Chest yaw / pitch relative to the pelvis heading (rad; pitch > 0 = forward) — event 2 Torso Archer.")]
        public float torsoYaw, torsoPitch;
        [Tooltip("Hand (forearm tip) target in the heading frame relative to the pelvis (m, MuJoCo axes) — event 4.")]
        public Vector3 handTarget;
        [Tooltip("Reaching arm: -1 left, +1 right, 0 none — event 4.")]
        public int arm;

        public enum Foot { None, Left, Right }

        public void ToArray(float[] a)
        {
            a[0] = pelvisHeight;
            a[1] = liftFoot == Foot.Left ? 1f : 0f;
            a[2] = liftFoot == Foot.Right ? 1f : 0f;
            a[3] = marchHz;
            a[4] = kneeLift;
            a[5] = torsoYaw;
            a[6] = torsoPitch;
            a[7] = handTarget.x;
            a[8] = handTarget.y;
            a[9] = handTarget.z;
            a[10] = arm;
        }

        public static SkillCommand FromArray(double[] a) => new()
        {
            pelvisHeight = (float)a[0],
            liftFoot = a[1] > 0.5 ? Foot.Left : a[2] > 0.5 ? Foot.Right : Foot.None,
            marchHz = (float)a[3],
            kneeLift = (float)a[4],
            torsoYaw = (float)a[5],
            torsoPitch = (float)a[6],
            handTarget = new Vector3((float)a[7], (float)a[8], (float)a[9]),
            arm = (int)Math.Round(a[10]),
        };
    }
}
