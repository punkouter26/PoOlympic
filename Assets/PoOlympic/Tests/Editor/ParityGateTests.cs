using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using PoOlympic.Editor;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Parity gates G2–G4 against every golden CPU-MuJoCo reference in parity/ (reference_trajectory_*.json).</summary>
    public class ParityGateTests
    {
        public static IEnumerable<string> References()
        {
            var dir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "parity"));
            foreach (var f in Directory.GetFiles(dir, "reference_trajectory_*.json"))
                yield return Path.GetFileNameWithoutExtension(f).Substring("reference_trajectory_".Length);
        }

        [OneTimeSetUp]
        public void Sync() => ParityHarness.SyncArtifacts();

        [TestCaseSource(nameof(References))]
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
