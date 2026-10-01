using UnityEditor;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Phase Z7 — the zombie's solo testbed (gates G0, G2–G5 on the zombie body): Testbed_Zombie.unity, built by the
    /// shared <see cref="SoloTestbed"/> (one scene per athlete body).
    /// </summary>
    public static class ZombieTestbed
    {
        public const string ScenePath = "Assets/PoOlympic/Scenes/Testbed_Zombie.unity";
        public const string SourceRelative = "training/assets/scene_zombie.xml";

        [MenuItem("PoOlympic/Build Testbed_Zombie (solo zombie)")]
        public static string Build() => SoloTestbed.Build("zombie");

        /// <summary>Open Testbed_Zombie with `brainFile` on the athlete, the standard (Froude-scaled) parity script and a
        /// 5 s recording → parity/unity_run_&lt;recordName&gt;.json (compare: tools/compare_closed_loop.py).</summary>
        public static string ConfigurePlay(string brainFile, string recordName) => SoloTestbed.ConfigurePlay("zombie", brainFile, recordName);
    }
}
