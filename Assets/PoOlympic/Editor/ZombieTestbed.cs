using System;
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Phase Z7 — the zombie's solo testbed (gates G0, G2–G5 on the zombie body): Testbed_Zombie.unity = the zombie's
    /// training scene (training/assets/scene_zombie.xml: athlete + 4-cube pool) with the camera / light / track / HUD look
    /// of Testbed_ZeroBrain, one PolicyRunner on the zombie contract and the bound zombie visual. ParityHarness runs every
    /// zombie reference (meta.body = "zombie") here; ConfigurePlay arms the standard parity script + recording for G5.
    /// </summary>
    public static class ZombieTestbed
    {
        public const string ScenePath = "Assets/PoOlympic/Scenes/Testbed_Zombie.unity";
        public const string SourceRelative = "training/assets/scene_zombie.xml";

        [MenuItem("PoOlympic/Build Testbed_Zombie (solo zombie)")]
        public static string Build()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var physics = AthleteImport.ImportIntoActiveScene(SourceRelative);

            var src = EditorSceneManager.OpenScene(ParityHarness.TestbedScene, OpenSceneMode.Additive);
            GameObject Copy(string name)
            {
                var original = Array.Find(src.GetRootGameObjects(), g => g.name == name) ?? throw new MissingReferenceException(name);
                var clone = UnityEngine.Object.Instantiate(original);
                clone.name = name;
                SceneManager.MoveGameObjectToScene(clone, scene);
                return clone;
            }
            var cam = Copy("Main Camera");
            Copy("Sun");
            Copy("TrackVisual");
            var hudGo = Copy("TestbedHUD");
            EditorSceneManager.CloseScene(src, true);
            SceneManager.SetActiveScene(scene);

            var cubeMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/PoOlympic/Materials/PoolCube.mat");
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
            {
                if (rend.GetComponent<MjGeom>() == null) continue;
                bool cube = rend.gameObject.name.StartsWith("cube");
                rend.enabled = cube;
                if (cube && cubeMat != null) rend.sharedMaterial = cubeMat;
            }

            var (contractPath, visualPath, _) = EventScenes.BodyAssets("zombie");
            var go = new GameObject("Athlete_Lane0");
            var r = go.AddComponent<PolicyRunner>();
            r.contractJson = AssetDatabase.LoadAssetAtPath<TextAsset>(contractPath);
            r.holdDefaultPose = true;             // until ConfigurePlay assigns a brain
            r.useStandardParityScript = true;
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(visualPath), go.transform);
            visual.name = "ZOMBIE_Visual";
            visual.transform.SetPositionAndRotation(Vector3.zero, VisualBinding.GltfToPlugin);
            var binder = visual.AddComponent<BoneBinder>();
            binder.Capture(physics.transform, "");
            var (pe, re) = binder.BindError();
            if (pe > 0.01f || re > 1f) throw new InvalidOperationException($"zombie bind error {pe * 1000f:F2} mm / {re:F2}°");

            var pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            pool.poolSize = 4;
            pool.runner = r;
            var bc = cam.GetComponent<BroadcastCamera>();
            bc.target = Array.Find(physics.GetComponentsInChildren<MjBody>(true), b => b.name == "pelvis");
            const float lambda = 0.618f;          // zombie height / MATT height: same framing at the zombie's size
            bc.offset *= lambda;
            bc.lookHeight *= lambda;
            var hud = hudGo.GetComponent<TestbedHud>();
            hud.runner = r;
            hud.lanes = new[] { r };
            hud.cubes = pool;
            hud.title = "PoOlympics — Testbed · ZOMBIE";
            hud.version = "v0 · zombie · G0/G2–G5";
            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"[ZombieTestbed] built {ScenePath} (bind {pe * 1000f:F2} mm / {re:F2}°)");
            return $"{ScenePath}: bind error {pe * 1000f:F2} mm / {re:F2}°";
        }

        /// <summary>Open Testbed_Zombie with `brainFile` on the athlete, the standard (Froude-scaled) parity script and a
        /// 5 s recording → parity/unity_run_&lt;recordName&gt;.json (compare: tools/compare_closed_loop.py).</summary>
        public static string ConfigurePlay(string brainFile, string recordName)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var r = UnityEngine.Object.FindFirstObjectByType<PolicyRunner>();
            r.brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}")
                      ?? throw new System.IO.FileNotFoundException(brainFile);
            r.brainSidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}.json");
            r.holdDefaultPose = false;
            r.useStandardParityScript = true;
            r.command = Vector3.zero;
            r.recordName = recordName;
            r.recordTicks = 250;
            EditorUtility.SetDirty(r);
            EditorSceneManager.SaveOpenScenes();
            return $"Testbed_Zombie: brain {brainFile}, recording {recordName}";
        }
    }
}
