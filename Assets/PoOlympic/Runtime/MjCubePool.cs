using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Pre-allocated pool of MuJoCo free-body cubes (cube0 … cubeN-1 in the MJCF, parked at x = 50 + 2i). Firing a
    /// cube teleports it by writing qpos/qvel through PolicyRunner at the next control tick — no Instantiate/Destroy,
    /// no PhysX. Positions/velocities are given in MuJoCo world coordinates (x forward, y left, z up).
    /// </summary>
    public class MjCubePool : MonoBehaviour
    {
        public PolicyRunner runner;
        public int poolSize = 4;
        public float launchSpeed = 6f;
        int _next;

        public string JointName(int i) => $"cube{i}_free";

        /// <summary>Launch the next pooled cube from `fromMj` towards `targetMj` (MuJoCo world coordinates).</summary>
        public void FireAt(Vector3 fromMj, Vector3 targetMj)
        {
            var dir = (targetMj - fromMj).normalized;
            Fire(fromMj, dir * launchSpeed);
        }

        public void Fire(Vector3 posMj, Vector3 velMj) => Fire(runner, posMj, velMj);

        /// <summary>Teleport the next pooled cube; applied at `via`'s next control tick (every lane ticks together).</summary>
        public void Fire(PolicyRunner via, Vector3 posMj, Vector3 velMj)
        {
            int i = _next;
            _next = (_next + 1) % poolSize;
            via.Request(new Disturbance
            {
                kind = "cube", target = JointName(i),
                qpos = new double[] { posMj.x, posMj.y, posMj.z, 1, 0, 0, 0 },
                qvel = new double[] { velMj.x, velMj.y, velMj.z, 0, 0, 0 },
            });
        }

        /// <summary>Drop a cube from `height` metres above the athlete's pelvis (2 kg, as in the robustness curriculum).</summary>
        public void DropOnAthlete(float height = 1.5f) => DropOnAthlete(runner, height);

        public unsafe void DropOnAthlete(PolicyRunner athlete, float height = 1.5f)
        {
            if (!athlete.Initialized) return;
            var d = MjScene.Instance.Data;
            int r = athlete.Binding.RootQposAdr;
            var p = new Vector3((float)d->qpos[r], (float)d->qpos[r + 1] + 0.2f, (float)d->qpos[r + 2] + 0.6f + height);
            Fire(athlete, p, Vector3.zero);
        }

        /// <summary>Horizontal shove: instantaneous Δv on the pelvis free joint (MuJoCo world frame).</summary>
        public void Shove(Vector2 dvXY) => Shove(runner, dvXY);

        public void Shove(PolicyRunner athlete, Vector2 dvXY) =>
            athlete.Request(new Disturbance { kind = "shove", target = "root", dqvel = new double[] { dvXY.x, dvXY.y, 0 } });
    }
}
