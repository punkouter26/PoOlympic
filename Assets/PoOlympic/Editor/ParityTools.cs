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
        public static string DumpFingerprint() => DumpFingerprint("fingerprint_unity.json", "", "cube0");

        /// <summary>G0 for one lane of a meet scene: athlete `prefix` + pool cube `cube` → parity/&lt;file&gt;.</summary>
        public static unsafe string DumpFingerprint(string file, string prefix, string cube)
        {
            // The plug-in's MjScene singleton only registers itself in Awake (Play mode). In Edit mode we use a
            // temporary instance and remove it afterwards so it never gets saved into the scene.
            foreach (var stale in Object.FindObjectsByType<MjScene>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                Object.DestroyImmediate(stale.gameObject);
            var scene = MjScene.Instance;
            scene.CreateScene();
            try
            {
                var json = ModelFingerprint.Dump(scene.Model, prefix, cube);
                Directory.CreateDirectory(ParityDir);
                var path = Path.Combine(ParityDir, file);
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

        /// <summary>
        /// Deterministic Play-mode stepping for unattended parity runs: the Editor does not tick the Play-mode
        /// player loop while unfocused, so we pause and advance with EditorApplication.Step() from the editor update
        /// loop, with a fixed capture frame time. Physics still advances exactly one mj_step per FixedUpdate.
        /// </summary>
        [MenuItem("PoOlympic/Parity/Drive Play Mode (deterministic stepping)")]
        public static void ArmDeterministicStepping()
        {
            if (!Application.isPlaying) { Debug.LogWarning("[Parity] enter Play mode first"); return; }
            Time.captureDeltaTime = 0.02f;
            EditorApplication.isPaused = true;
            EditorApplication.CallbackFunction step = null;
            step = () =>
            {
                if (!Application.isPlaying) { EditorApplication.update -= step; return; }
                EditorApplication.Step();
            };
            EditorApplication.update += step;
        }
    
        /// <summary>Static event props (world geoms named `prefix`*) of the running scene's compiled model as JSON
        /// {name: {pos, size, contype, conaffinity}} — G0 covers each athlete lane, this covers rails / poles.</summary>
        public static unsafe string DumpProps(string prefix)
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null) return "{}";
            var m = MjScene.Instance.Model;
            var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);
            var sb = new System.Text.StringBuilder("{");
            foreach (var (name, id) in geoms)
            {
                if (!name.StartsWith(prefix)) continue;
                if (sb.Length > 1) sb.Append(',');
                string V(double* a) => $"[{a[3 * id].ToString("R", System.Globalization.CultureInfo.InvariantCulture)},{a[3 * id + 1].ToString("R", System.Globalization.CultureInfo.InvariantCulture)},{a[3 * id + 2].ToString("R", System.Globalization.CultureInfo.InvariantCulture)}]";
                sb.Append($"\"{name}\":{{\"pos\":{V(m->geom_pos)},\"size\":{V(m->geom_size)},\"contype\":{m->geom_contype[id]},\"conaffinity\":{m->geom_conaffinity[id]}}}");
            }
            return sb.Append('}').ToString();
        }

        /// <summary>Deterministically step Play mode until the first PolicyRunner reaches `tick`, then stay paused.</summary>
        public static void StepUntilTick(int tick)
        {
            if (!Application.isPlaying) { Debug.LogWarning("[Parity] enter Play mode first"); return; }
            var runner = Object.FindFirstObjectByType<PolicyRunner>();
            Time.captureDeltaTime = 0.02f;
            EditorApplication.isPaused = true;
            EditorApplication.CallbackFunction step = null;
            step = () =>
            {
                if (!Application.isPlaying || runner == null || runner.ControlTick >= tick) { EditorApplication.update -= step; return; }
                EditorApplication.Step();
            };
            EditorApplication.update += step;
        }
    }
}
