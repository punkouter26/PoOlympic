using Mujoco;
using UnityEditor;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>
    /// B5 — Place MATT's glTF visual onto the imported physics body and capture bone offsets.
    /// glTFast puts the character facing Unity +Z (left = −X); the MuJoCo plug-in puts MuJoCo +X (forward) on
    /// Unity +X and MuJoCo +Y (left) on Unity +Z. Both are proper rotations of glTF space, related by a +90° yaw.
    /// </summary>
    public static class VisualBinding
    {
        public const string MattAsset = "Assets/PoOlympic/Art/MATT.glb";
        public static readonly Quaternion GltfToPlugin = Quaternion.Euler(0f, 90f, 0f);

        [MenuItem("PoOlympic/Bind MATT Visual To Physics")]
        public static BoneBinder Bind()
        {
            var physics = GameObject.Find(AthleteImport.SceneRootName) ?? throw new System.InvalidOperationException("import the athlete MJCF first");
            var lane = GameObject.Find("Athlete_Lane0") ?? new GameObject("Athlete_Lane0");
            var old = lane.transform.Find("MATT_Visual");
            if (old != null) Object.DestroyImmediate(old.gameObject);

            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(MattAsset);
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(prefab, lane.transform);
            visual.name = "MATT_Visual";
            visual.transform.SetPositionAndRotation(Vector3.zero, GltfToPlugin);

            // Physics geoms are for simulation only — hide any renderers the importer created.
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
                if (rend.GetComponent<MjGeom>() != null && rend.gameObject.name != "ground" && !rend.gameObject.name.StartsWith("cube")) rend.enabled = false;

            var binder = visual.GetComponent<BoneBinder>() ?? visual.AddComponent<BoneBinder>();
            binder.Capture(physics.transform);
            var (p, r) = binder.BindError();
            Debug.Log($"[VisualBinding] {binder.bindings.Count} bones bound; bind error {p * 1000f:F2} mm, body frame rotation {r:F3}°");
            EditorUtility.SetDirty(binder);
            return binder;
        }
    }
}
