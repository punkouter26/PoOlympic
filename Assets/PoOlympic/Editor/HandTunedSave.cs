using System.Collections.Generic;
using System.IO;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Save a freshly built scene over its old file, carrying over every object the old file marked <see cref="HandTuned"/>
    /// (matched by hierarchy path): its local transform and the marker. Objects the new build no longer has are reported.
    /// </summary>
    public static class HandTunedSave
    {
        public static bool SaveBuilt(Scene scene, string path)
        {
            if (File.Exists(path))
            {
                var tuned = new List<(string path, Vector3 pos, Quaternion rot, Vector3 scale)>();
                var old = EditorSceneManager.OpenScene(path, OpenSceneMode.Additive);
                foreach (var root in old.GetRootGameObjects())
                    foreach (var h in root.GetComponentsInChildren<HandTuned>(true))
                        tuned.Add((PathOf(h.transform), h.transform.localPosition, h.transform.localRotation, h.transform.localScale));
                EditorSceneManager.CloseScene(old, true);
                SceneManager.SetActiveScene(scene);
                foreach (var (p, pos, rot, scale) in tuned)
                {
                    var t = Find(scene, p);
                    if (t == null) { Debug.LogWarning($"[HandTuned] {path}: {p} is not in the new build — hand placement dropped"); continue; }
                    t.SetLocalPositionAndRotation(pos, rot);
                    t.localScale = scale;
                    if (t.GetComponent<HandTuned>() == null) t.gameObject.AddComponent<HandTuned>();
                }
                if (tuned.Count > 0) Debug.Log($"[HandTuned] {path}: {tuned.Count} hand-placed object(s) kept");
            }
            return EditorSceneManager.SaveScene(scene, path);
        }

        static string PathOf(Transform t) => t.parent == null ? t.name : PathOf(t.parent) + "/" + t.name;

        static Transform Find(Scene scene, string path)
        {
            var parts = path.Split('/');
            foreach (var root in scene.GetRootGameObjects())
            {
                if (root.name != parts[0]) continue;
                var t = parts.Length == 1 ? root.transform : root.transform.Find(string.Join("/", parts, 1, parts.Length - 1));
                if (t != null) return t;
            }
            return null;
        }
    }
}
