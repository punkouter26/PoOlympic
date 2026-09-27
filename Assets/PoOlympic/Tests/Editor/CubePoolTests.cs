using Mujoco;
using NUnit.Framework;
using PoOlympic.Editor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>B7 — a pooled MuJoCo cube fired at MATT strikes him through MuJoCo contacts only.</summary>
    public unsafe class CubePoolTests
    {
        [Test]
        public void FiredPoolCubeHitsAthleteThroughMuJoCo()
        {
            EditorSceneManager.OpenScene(ParityHarness.TestbedScene, OpenSceneMode.Single);
            foreach (var s in Object.FindObjectsByType<MjScene>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                Object.DestroyImmediate(s.gameObject);
            var scene = MjScene.Instance;
            scene.CreateScene();
            try
            {
                var m = scene.Model;
                var d = scene.Data;
                var contract = ParityHarness.LoadContract();
                var bind = new AthleteBinding(m, contract);
                var joints = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, (int)m->njnt);
                var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);
                int cubeGeom = geoms["cube0_geom"];

                // default standing state, PD holding the default pose
                MujocoLib.mj_resetData(m, d);
                foreach (var jq in contract.default_joint_qpos)
                    for (int i = 0; i < jq.qpos.Length; i++) d->qpos[m->jnt_qposadr[joints[jq.joint]] + i] = jq.qpos[i];
                MujocoLib.mj_forward(m, d);
                for (int i = 0; i < bind.ActuatorIds.Length; i++) d->ctrl[bind.ActuatorIds[i]] = bind.DefaultPos[i];

                // fire cube0 from 1.5 m in front of the chest, straight at it (MuJoCo frame: +x forward)
                new Disturbance { kind = "cube", target = "cube0_free",
                                  qpos = new[] { 1.5, 0.0, 1.3, 1, 0, 0, 0 }, qvel = new[] { -8.0, 0, 0, 0, 0, 0 } }
                    .Apply(m, d, joints, "");

                bool hitAthlete = false;
                double minPelvisVx = 0;
                for (int k = 0; k < 100; k++) // 0.5 s
                {
                    MujocoLib.mj_step(m, d);
                    for (int c = 0; c < d->ncon; c++)
                    {
                        var con = d->contact[c];
                        int other = con.geom1 == cubeGeom ? con.geom2 : con.geom2 == cubeGeom ? con.geom1 : -1;
                        if (other >= 0 && m->geom_bodyid[other] != 0) hitAthlete = true;
                    }
                    minPelvisVx = System.Math.Min(minPelvisVx, d->qvel[bind.RootDofAdr]);
                }
                Assert.IsTrue(hitAthlete, "cube never touched an athlete geom");
                Assert.Less(minPelvisVx, -0.05, "impact did not push the athlete backwards");
            }
            finally
            {
                scene.DestroyScene();
                Object.DestroyImmediate(scene.gameObject);
            }
        }
    }
}
