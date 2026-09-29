using System.IO;
using NUnit.Framework;
using PoOlympic.Editor;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Contract v4 (stance-skill block): mirrors training/tests/test_contract.py (layout, clock, block order).</summary>
    public class ContractV4Tests
    {
        [OneTimeSetUp]
        public void Sync() => ParityHarness.SyncArtifacts();    // parity/contract.json → Models (as ParityGateTests)

        static Contract Load() => ParityHarness.LoadContract();

        [Test]
        public void SkillBlockFollowsTheV3Obs()
        {
            var c = Load();
            Assert.IsTrue(c.SupportsSkills, "contract.json has no v4 skill_block");
            Assert.AreEqual(84, c.ObsDimFor(3));
            Assert.AreEqual(95, c.ObsDimFor(4));
            Assert.AreEqual(c.obs_dim, c.skill_block.offset);
        }

        [Test]
        public void MarchCadenceDrivesTheClock()
        {
            var c = Load();
            Assert.AreEqual(1.5 * c.decimation * c.timestep, c.AdvancePhase(0.0, Vector3.zero, 1.5), 1e-12);
            Assert.AreEqual(0.0, c.AdvancePhase(0.3, Vector3.zero, 0.0));                        // v3: frozen at zero command
            Assert.AreEqual(c.AdvancePhase(0.1, Vector3.right), c.AdvancePhase(0.1, Vector3.right, 0.0));
        }

        [Test]
        public void SkillCommandOrderMatchesPython()
        {
            // contract.py SkillCommand(pelvis_height=-0.3, lift_foot="r", march_hz=1.5, knee_lift=0.2, torso_yaw=0.4,
            //                          torso_pitch=-0.1, hand=(0.5, -0.2, 0.3), arm=1).to_array()
            var s = new SkillCommand
            {
                pelvisHeight = -0.3f, liftFoot = SkillCommand.Foot.Right, marchHz = 1.5f, kneeLift = 0.2f,
                torsoYaw = 0.4f, torsoPitch = -0.1f, handTarget = new Vector3(0.5f, -0.2f, 0.3f), arm = 1,
            };
            var a = new float[ObservationBuilder.SkillDim];
            s.ToArray(a);
            CollectionAssert.AreEqual(new[] { -0.3f, 0f, 1f, 1.5f, 0.2f, 0.4f, -0.1f, 0.5f, -0.2f, 0.3f, 1f }, a);
            Assert.AreEqual(s, SkillCommand.FromArray(System.Array.ConvertAll(a, x => (double)x)));
        }

        [Test]
        public void V4TestBrainSidecarIsVersion4()
        {
            var path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "parity", "brains", "random_brain_v4.onnx.json"));
            var sc = JsonUtility.FromJson<BrainSidecar>(File.ReadAllText(path));
            Assert.AreEqual("4", sc.contract_version);
        }
    }
}
