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
    /// The solo testbed of one athlete body (gates G0, G2–G5 on that body): Testbed_&lt;Body&gt;.unity = the body's training
    /// scene (training/assets/scene_&lt;body&gt;.xml: athlete + 4-cube pool) with the camera / light / track / HUD look of
    /// Testbed_ZeroBrain, one PolicyRunner on the body's contract and its bound visual. ParityHarness runs every
    /// reference of that body (meta.body) here; ConfigurePlay arms the standard parity script + recording for G5.
    /// Bodies: zombie (Phase Z7), grandma (Phase G4).
    /// </summary>
    public static class SoloTestbed
    {
        public static string ScenePathOf(string body) =>
            $"Assets/PoOlympic/Scenes/Testbed_{char.ToUpperInvariant(body[0])}{body.Substring(1)}.unity";

        public static string SourceOf(string body) => $"training/assets/scene_{body}.xml";

        [MenuItem("PoOlympic/Build Testbed_Grandma (solo grandma)")]
        public static string BuildGrandma() => Build("grandma");

        public static string Build(string body)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var physics = AthleteImport.ImportIntoActiveScene(SourceOf(body));

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

            string upper = body.ToUpperInvariant();
            var (contractPath, visualPath, _) = EventScenes.BodyAssets(body);
            var go = new GameObject("Athlete_Lane0");
            var r = go.AddComponent<PolicyRunner>();
            r.contractJson = AssetDatabase.LoadAssetAtPath<TextAsset>(contractPath) ?? throw new System.IO.FileNotFoundException(contractPath);
            r.holdDefaultPose = true;             // until ConfigurePlay assigns a brain
            r.useStandardParityScript = true;
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(visualPath) ?? throw new System.IO.FileNotFoundException(visualPath);
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(prefab, go.transform);
            visual.name = $"{upper}_Visual";
            visual.transform.SetPositionAndRotation(Vector3.zero, VisualBinding.GltfToPlugin);
            var binder = visual.AddComponent<BoneBinder>();
            binder.Capture(physics.transform, "");
            var (pe, re) = binder.BindError();
            if (pe > 0.01f || re > 1f) throw new InvalidOperationException($"{body} bind error {pe * 1000f:F2} mm / {re:F2}°");

            var pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            pool.poolSize = 4;
            pool.runner = r;
            var bc = cam.GetComponent<BroadcastCamera>();
            bc.target = Array.Find(physics.GetComponentsInChildren<MjBody>(true), b => b.name == "pelvis");
            double s = Contract.Parse(r.contractJson.text).SpeedScale;
            float lambda = (float)(s * s);        // body height / MATT height: same framing at the body's size
            bc.offset *= lambda;
            bc.lookHeight *= lambda;
            var hud = hudGo.GetComponent<TestbedHud>();
            hud.runner = r;
            hud.lanes = new[] { r };
            hud.cubes = pool;
            hud.title = $"PoOlympics — Testbed · {upper}";
            hud.version = $"v0 · {body} · G0/G2–G5";
            string path = ScenePathOf(body);
            EditorSceneManager.SaveScene(scene, path);
            Debug.Log($"[SoloTestbed] built {path} (bind {pe * 1000f:F2} mm / {re:F2}°, {binder.bindings.Count} bones)");
            return $"{path}: bind error {pe * 1000f:F2} mm / {re:F2}°, {binder.bindings.Count} bones";
        }

        /// <summary>Open the body's testbed with `brainFile` on the athlete, the standard (Froude-scaled) parity script and
        /// a 5 s recording → parity/unity_run_&lt;recordName&gt;.json (compare: tools/compare_closed_loop.py).</summary>
        public static string ConfigurePlay(string body, string brainFile, string recordName)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            EditorSceneManager.OpenScene(ScenePathOf(body), OpenSceneMode.Single);
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
            return $"{System.IO.Path.GetFileNameWithoutExtension(ScenePathOf(body))}: brain {brainFile}, recording {recordName}";
        }
    }
}
