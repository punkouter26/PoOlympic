using System;
using System.IO;
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Phase D — authors the event scenes in-editor from the generated MJCFs. Anything the athlete can touch is a MuJoCo
    /// geom from training/assets/*.xml; Unity-side art (stadium, dressing) is render-only.
    /// </summary>
    public static class EventScenes
    {
        public const string IronPedestalScene = "Assets/PoOlympic/Scenes/Event_IronPedestal.unity";
        public const string PedestalSource = "training/assets/scene_pedestal.xml";
        public const string IronMaterial = "Assets/PoOlympic/Materials/IronPedestal.mat";
        public const string DefaultRung0Brain = "r0_v2_it1000.onnx";
        public const string StadiumAsset = "Assets/PoOlympic/Art/Stadium/Stadium.glb";

        /// <summary>
        /// Render-only stadium (SourceArt/Stadium/stadium.blend → Stadium.glb, authored in MuJoCo axes). glTF→glTFast maps
        /// Blender (x, y, z) to Unity (−x, z, −y); a 180° yaw makes it (x, z, y) = the MuJoCo plug-in's mapping. The root is
        /// then shifted so the named venue anchor (VENUE_*) sits on the MuJoCo origin with the infield at `groundY`.
        /// </summary>
        public static GameObject PlaceStadium(Scene scene, string venue, float groundY)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(StadiumAsset) ?? throw new FileNotFoundException(StadiumAsset);
            var st = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            st.name = "Stadium";
            st.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 180f, 0f));
            var anchor = Array.Find(st.GetComponentsInChildren<Transform>(true), t => t.name == venue)
                         ?? throw new MissingReferenceException($"venue anchor {venue} not in {StadiumAsset}");
            var p = anchor.position;
            st.transform.position = new Vector3(-p.x, groundY, -p.z);
            foreach (var c in st.GetComponentsInChildren<Component>(true))
                if (c is Collider || c is Rigidbody) throw new InvalidOperationException($"stadium must be render-only: {c.GetType().Name} on {c.name}");
            return st;
        }

        static Material IronPedestalMaterial()
        {
            var mat = AssetDatabase.LoadAssetAtPath<Material>(IronMaterial);
            if (mat != null) return mat;
            mat = new Material(Shader.Find("Universal Render Pipeline/Lit")) { name = "IronPedestal" };
            mat.SetColor("_BaseColor", new Color(0.23f, 0.24f, 0.26f));
            mat.SetFloat("_Metallic", 0.85f);
            mat.SetFloat("_Smoothness", 0.45f);
            AssetDatabase.CreateAsset(mat, IronMaterial);
            return mat;
        }

        [MenuItem("PoOlympic/Events/Build Event 1 — Iron Pedestal")]
        public static string BuildIronPedestal() => BuildIronPedestal(DefaultRung0Brain);

        public static string BuildIronPedestal(string brainFile)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var physics = AthleteImport.ImportIntoActiveScene(PedestalSource);

            // camera / light / render-only track from the testbed look
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
            EditorSceneManager.CloseScene(src, true);
            SceneManager.SetActiveScene(scene);
            PlaceStadium(scene, "VENUE_CentreStage", -0.5f); // the MuJoCo ground sits at z = -0.5 in this scene

            // MuJoCo geom renderers: only the pedestal and the pool cubes are shown
            var cubeMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/PoOlympic/Materials/PoolCube.mat");
            var iron = IronPedestalMaterial();
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
            {
                if (rend.GetComponent<MjGeom>() == null) continue;
                var n = rend.gameObject.name;
                rend.enabled = n == "pedestal" || n.StartsWith("cube");
                if (n == "pedestal") rend.sharedMaterial = iron;
                else if (n.StartsWith("cube") && cubeMat != null) rend.sharedMaterial = cubeMat;
            }

            // athlete: runner + pooled cubes + bound MATT visual
            var athlete = new GameObject("Athlete");
            var runner = athlete.AddComponent<PolicyRunner>();
            runner.contractJson = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/PoOlympic/Models/contract.json");
            runner.brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}")
                           ?? throw new FileNotFoundException(brainFile);
            runner.brainSidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}.json");
            runner.useStandardParityScript = false;
            var pool = athlete.AddComponent<MjCubePool>();
            pool.runner = runner;
            pool.poolSize = 4;
            var visual = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(VisualBinding.MattAsset), athlete.transform);
            visual.name = "MATT_Visual";
            visual.transform.SetPositionAndRotation(Vector3.zero, VisualBinding.GltfToPlugin);
            var binder = visual.AddComponent<BoneBinder>();
            binder.Capture(physics.transform);

            // event + HUD
            var evGo = new GameObject("IronPedestalEvent");
            var ev = evGo.AddComponent<IronPedestalEvent>();
            ev.runner = runner;
            ev.cubes = pool;
            var hud = new GameObject("EventHUD").AddComponent<EventHud>();
            hud.ev = ev;
            hud.version = $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}";

            var bc = cam.GetComponent<BroadcastCamera>();
            bc.target = Array.Find(physics.GetComponentsInChildren<MjBody>(true), b => b.name == "pelvis");
            bc.offset = new Vector3(1.6f, 0.6f, -4.6f); // three-quarter view: the pedestal edges read on camera

            EditorSceneManager.SaveScene(scene, IronPedestalScene);
            var (p, rot) = binder.BindError();
            return $"{IronPedestalScene}: brain {brainFile}, bind error {p * 1000f:F2} mm / {rot:F3}°";
        }
    }
}
