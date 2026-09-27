using NUnit.Framework;
using PoOlympic.Editor;

namespace PoOlympic.Tests
{
    /// <summary>Phase B parity gates G2–G4 against the golden CPU-MuJoCo references (parity/reference_trajectory_*.json).</summary>
    public class ParityGateTests
    {
        [OneTimeSetUp]
        public void Sync() => ParityHarness.SyncArtifacts();

        [TestCase("zero")]
        [TestCase("random")]
        public void Gates_G2_G3_G4(string reference)
        {
            var r = ParityHarness.Run(reference);
            var summary = $"G2 obs {r.g2MaxAbs:E2} | G3 action {r.g3MaxAbsAction:E2} ctrl {r.g3MaxAbsCtrl:E2} | " +
                          $"G4 qpos 1s {r.g4MaxAbs1s:E2} 5s {r.g4MaxAbs5s:E2} (first tick over tol: {r.g4FirstTickOver})";
            TestContext.WriteLine(summary);
            Assert.IsTrue(r.G2, "G2 failed: " + summary);
            Assert.IsTrue(r.G3, "G3 failed: " + summary);
            Assert.IsTrue(r.G4, "G4 failed: " + summary);
        }
    }
}
