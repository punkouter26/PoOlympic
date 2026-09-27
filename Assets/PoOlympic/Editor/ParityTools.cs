using System.IO;
using Mujoco;
using UnityEditor;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>Editor entry points for the parity gates (DESIGN.md §4).</summary>
    public static class ParityTools
    {
        public static string ParityDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "parity"));

        /// <summary>G0 — compile the open scene's MuJoCo model and write parity/fingerprint_unity.json.</summary>
        [MenuItem("PoOlympic/Parity/G0 Dump Unity Fingerprint")]
        public static unsafe string DumpFingerprint()
        {
            // The plug-in's MjScene singleton only registers itself in Awake (Play mode). In Edit mode we use a
            // temporary instance and remove it afterwards so it never gets saved into the scene.
            foreach (var stale in Object.FindObjectsByType<MjScene>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                Object.DestroyImmediate(stale.gameObject);
            var scene = MjScene.Instance;
            scene.CreateScene();
            try
            {
                var json = ModelFingerprint.Dump(scene.Model);
                Directory.CreateDirectory(ParityDir);
                var path = Path.Combine(ParityDir, "fingerprint_unity.json");
                File.WriteAllText(path, json);
                Debug.Log($"[Parity] wrote {path}");
                return path;
            }
            finally
            {
                scene.DestroyScene();
                Object.DestroyImmediate(scene.gameObject);
            }
        }
    }
}
