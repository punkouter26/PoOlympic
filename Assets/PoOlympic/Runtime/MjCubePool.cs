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

        int[] _qposAdr, _dofAdr;
        System.IntPtr _cachedFor;   // the model the addresses belong to: play starts without a scene reload, so a new model (lineup) must re-read them

        /// <summary>The fastest cube right now (Unity world position, m/s); parked cubes rest at 0. Read only.</summary>
        public unsafe bool Fastest(out Vector3 position, out float speed)
        {
            position = default;
            speed = 0f;
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null) return false;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            if (_qposAdr == null || _qposAdr.Length != poolSize || _cachedFor != (System.IntPtr)m)
            {
                var joints = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, (int)m->njnt);
                _qposAdr = new int[poolSize];
                _dofAdr = new int[poolSize];
                _cachedFor = (System.IntPtr)m;
                for (int i = 0; i < poolSize; i++)
                {
                    bool found = joints.TryGetValue(JointName(i), out int j);
                    _qposAdr[i] = found ? m->jnt_qposadr[j] : -1;
                    _dofAdr[i] = found ? m->jnt_dofadr[j] : -1;
                }
            }
            for (int i = 0; i < poolSize; i++)
            {
                int v = _dofAdr[i], q = _qposAdr[i];
                if (v < 0) continue;
                float s = Mathf.Sqrt((float)(d->qvel[v] * d->qvel[v] + d->qvel[v + 1] * d->qvel[v + 1] + d->qvel[v + 2] * d->qvel[v + 2]));
                if (s <= speed) continue;
                speed = s;
                position = new Vector3((float)d->qpos[q], (float)d->qpos[q + 2], (float)d->qpos[q + 1]);
            }
            return speed > 0f;
        }

        /// <summary>Horizontal shove: instantaneous Δv on the pelvis free joint (MuJoCo world frame).</summary>
        public void Shove(Vector2 dvXY) => Shove(runner, dvXY);

        public void Shove(PolicyRunner athlete, Vector2 dvXY) =>
            athlete.Request(new Disturbance { kind = "shove", target = "root", dqvel = new double[] { dvXY.x, dvXY.y, 0 } });
    }
}
