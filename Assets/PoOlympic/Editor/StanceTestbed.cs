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
    /// Rung S showcase: Testbed_Stance.unity = MATT on the mattbio body (training/assets/scene_mattbio.xml, the Rung S
    /// training body) with the Testbed_ZeroBrain camera / light / track / HUD, one contract v4 PolicyRunner and a
    /// StanceSkillDemo that cycles squat → flamingo → march → torso aim → reach. No parity script: the athlete only does
    /// what the skill command asks (HUD Shove / Drop cube still work).
    /// </summary>
    public static class StanceTestbed
    {
        public const string ScenePath = "Assets/PoOlympic/Scenes/Testbed_Stance.unity";
        public const string SourceRelative = "training/assets/scene_mattbio.xml";
        public const string ContractPath = "Assets/PoOlympic/Models/contract_mattbio.json";

        [MenuItem("PoOlympic/Build Testbed_Stance (mattbio, Rung S skills)")]
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

            var (_, visualPath, _) = EventScenes.BodyAssets("matt");     // mattbio = MATT's skeleton (same bodies)
            var go = new GameObject("Athlete_Lane0");
            var r = go.AddComponent<PolicyRunner>();
            r.contractJson = AssetDatabase.LoadAssetAtPath<TextAsset>(ContractPath) ?? throw new System.IO.FileNotFoundException(ContractPath);
            r.holdDefaultPose = true;             // until Configure assigns a brain
            r.useStandardParityScript = false;
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(visualPath), go.transform);
            visual.name = "MATT_Visual";
            visual.transform.SetPositionAndRotation(Vector3.zero, VisualBinding.GltfToPlugin);
            var binder = visual.AddComponent<BoneBinder>();
            binder.Capture(physics.transform, "");
            var (pe, re) = binder.BindError();
            if (pe > 0.01f || re > 1f) throw new InvalidOperationException($"mattbio bind error {pe * 1000f:F2} mm / {re:F2}°");

            var demo = go.AddComponent<StanceSkillDemo>();
            demo.runner = r;
            demo.steps = StanceSkillDemo.DefaultSteps();

            var pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            pool.poolSize = 4;
            pool.runner = r;
            cam.GetComponent<BroadcastCamera>().target = Array.Find(physics.GetComponentsInChildren<MjBody>(true), b => b.name == "pelvis");
            var hud = hudGo.GetComponent<TestbedHud>();
            hud.runner = r;
            hud.lanes = new[] { r };
            hud.cubes = pool;
            hud.title = "PoOlympics — Testbed · Stance skills";
            hud.version = "v0 · mattbio · Rung S";
            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"[StanceTestbed] built {ScenePath} (bind {pe * 1000f:F2} mm / {re:F2}°)");
            return $"{ScenePath}: bind error {pe * 1000f:F2} mm / {re:F2}°";
        }

        /// <summary>Open Testbed_Stance with `brainFile` (parity/brains → Models/Brains) on the athlete, skill cycling on.</summary>
        public static string Configure(string brainFile)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var r = UnityEngine.Object.FindFirstObjectByType<PolicyRunner>();
            r.brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}")
                      ?? throw new System.IO.FileNotFoundException(brainFile);
            r.brainSidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}.json");
            r.holdDefaultPose = false;
            r.useStandardParityScript = false;
            r.command = Vector3.zero;
            EditorUtility.SetDirty(r);
            EditorSceneManager.SaveOpenScenes();
            return $"Testbed_Stance: brain {brainFile}";
        }
    }
}
