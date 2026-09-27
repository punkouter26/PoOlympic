using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Static rules for runtime code (DESIGN.md §1): no runtime object churn, no PhysX API use.</summary>
    public class RuntimeRulesTests
    {
        static string[] RuntimeSources() =>
            Directory.GetFiles(Path.Combine(Application.dataPath, "PoOlympic", "Runtime"), "*.cs", SearchOption.AllDirectories);

        [Test]
        public void NoInstantiateOrDestroyInRuntimeCode()
        {
            var rx = new Regex(@"\b(Instantiate|Destroy|DestroyImmediate)\s*\(");
            foreach (var f in RuntimeSources())
                Assert.IsFalse(rx.IsMatch(File.ReadAllText(f)), $"{Path.GetFileName(f)} calls Instantiate/Destroy — use the pre-allocated MuJoCo pool");
        }

        [Test]
        public void NoPhysXApiInRuntimeCode()
        {
            var rx = new Regex(@"\b(Rigidbody|Collider|Physics\.(Raycast|Simulate|OverlapSphere|SphereCast))\b");
            foreach (var f in RuntimeSources())
                Assert.IsFalse(rx.IsMatch(File.ReadAllText(f)), $"{Path.GetFileName(f)} uses a PhysX API");
        }
    }
}
