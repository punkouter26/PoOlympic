using System;
using System.IO;
using System.Linq;
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
        public const int AthleteLane = 3;
        public const string StadiumAsset = "Assets/PoOlympic/Art/Stadium/Stadium.glb";

        public const string VenuesJson = "Assets/PoOlympic/Art/Stadium/venues.json";

        /// <summary>(position, yaw°) of one competitor spot from venues.json (MuJoCo axes; written by build_venues.py).</summary>
        public static (Vector3 posMj, float yawDeg) VenueLane(int eventNum, int lane)
        {
            var root = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(VenuesJson));
            var l = root["events"][$"{eventNum:00}"]["lanes"][lane];
            var p = l["pos"];
            return (new Vector3((float)p[0], (float)p[1], (float)p[2]), (float)l["yaw_deg"]);
        }

        /// <summary>
        /// Render-only stadium (SourceArt/Stadium/stadium.blend → Stadium.glb, authored in MuJoCo axes). glTF→glTFast maps
        /// Blender (x, y, z) to Unity (−x, z, −y); a 180° yaw makes it (x, z, y) = the MuJoCo plug-in's mapping. It is then
        /// turned about the vertical by the lane's yaw (MuJoCo yaw ψ ≙ Unity Euler(0, −ψ, 0), so undoing ψ is Euler(0, +ψ, 0))
        /// and shifted so the competitor spot E##_L# lands on the MuJoCo origin: the athlete (default pose facing +x) then
        /// stands exactly on that spot facing the event direction. Spot height = surface under the athlete (pedestal top).
        /// </summary>
        public static GameObject PlaceStadium(Scene scene, int eventNum, int lane)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(StadiumAsset) ?? throw new FileNotFoundException(StadiumAsset);
            var st = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            st.name = "Stadium";
            var (_, yaw) = VenueLane(eventNum, lane);
            st.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, yaw, 0f) * Quaternion.Euler(0f, 180f, 0f));
            var name = $"E{eventNum:00}_L{lane}";
            var anchor = Array.Find(st.GetComponentsInChildren<Transform>(true), t => t.name == name)
                         ?? throw new MissingReferenceException($"lane anchor {name} not in {StadiumAsset}");
            st.transform.position = -anchor.position;
            foreach (var c in st.GetComponentsInChildren<Component>(true))
                if (c is Collider || c is Rigidbody) throw new InvalidOperationException($"stadium must be render-only: {c.GetType().Name} on {c.name}");
            return st;
        }

        public const string HubScene = "Assets/PoOlympic/Scenes/Stadium_Hub.unity";
        public const string CatalogJson = "Assets/PoOlympic/Art/Stadium/events_catalog.json";

        /// <summary>Event number → built event scene (grows as events are implemented).</summary>
        public static readonly System.Collections.Generic.Dictionary<int, string> EventScenePaths = new()
        {
            { 1, IronPedestalScene },
        };

        /// <summary>
        /// Stadium hub: the stadium in the MuJoCo frame (yaw 180° only) with all 30 events as EventVenue objects (catalogue
        /// data + their 8 competitor anchors), an event picker, and Build Settings = hub + every built event scene.
        /// </summary>
        [MenuItem("PoOlympic/Events/Build Stadium Hub (all 30 events)")]
        public static string BuildStadiumHub()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(StadiumAsset) ?? throw new FileNotFoundException(StadiumAsset);
            var st = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            st.name = "Stadium";
            st.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, 180f, 0f)); // stadium coords = MuJoCo coords
            var anchors = st.GetComponentsInChildren<Transform>(true);

            var src = EditorSceneManager.OpenScene(ParityHarness.TestbedScene, OpenSceneMode.Additive);
            GameObject Copy(string name)
            {
                var clone = UnityEngine.Object.Instantiate(Array.Find(src.GetRootGameObjects(), g => g.name == name));
                clone.name = name;
                SceneManager.MoveGameObjectToScene(clone, scene);
                return clone;
            }
            var cam = Copy("Main Camera");
            Copy("Sun");
            EditorSceneManager.CloseScene(src, true);
            SceneManager.SetActiveScene(scene);
            cam.GetComponent<BroadcastCamera>().target = null; // letterbox only; the director drives the camera
            cam.GetComponent<Camera>().farClipPlane = 1500f;
            cam.GetComponent<Camera>().nearClipPlane = 0.5f; // overview camera: never closer than a few metres

            var catalog = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(CatalogJson))["events"];
            var root = new GameObject("Events");
            var venues = new System.Collections.Generic.List<EventVenue>();
            foreach (var e in catalog)
            {
                int n = (int)e["number"];
                var go = new GameObject($"E{n:00} {(string)e["name"]}");
                go.transform.SetParent(root.transform);
                var v = go.AddComponent<EventVenue>();
                v.number = n;
                v.eventName = (string)e["name"];
                v.phase = (int)e["phase"];
                v.skill = (string)e["skill"];
                v.brain = (string)e["brain"];
                v.rules = (string)e["rules"];
                v.eventScene = EventScenePaths.TryGetValue(n, out var path) ? Path.GetFileNameWithoutExtension(path) : "";
                for (int k = 0; k < 8; k++)
                    v.lanes[k] = Array.Find(anchors, t => t.name == $"E{n:00}_L{k}") ?? throw new MissingReferenceException($"E{n:00}_L{k}");
                go.transform.position = v.Centre;
                venues.Add(v);
            }
            if (venues.Count != 30) throw new InvalidOperationException($"catalogue has {venues.Count} events, expected 30");
            var director = new GameObject("StadiumDirector").AddComponent<StadiumDirector>();
            director.cam = cam.GetComponent<Camera>();
            director.venues = venues.ToArray();
            director.selected = 1;
            EditorSceneManager.SaveScene(scene, HubScene);

            var list = new System.Collections.Generic.List<EditorBuildSettingsScene> { new(HubScene, true) };
            foreach (var path in EventScenePaths.Values) list.Add(new EditorBuildSettingsScene(path, true));
            EditorBuildSettings.scenes = list.ToArray();
            return $"{HubScene}: {venues.Count} event venues, {venues.Count(x => x.Playable)} playable; build scenes {list.Count}";
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
            // Event 1 venue: 8 pedestals; this single athlete takes lane 4 (E01_L3). The stadium's pedestal there is the
            // visual of the MuJoCo pedestal geom (same 1 x 1 x 0.5 m box, top at z = 0); the other 7 wait for D4.
            PlaceStadium(scene, 1, AthleteLane);

            // MuJoCo geom renderers: only the pedestal and the pool cubes are shown
            var cubeMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/PoOlympic/Materials/PoolCube.mat");
            var iron = IronPedestalMaterial();
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
            {
                if (rend.GetComponent<MjGeom>() == null) continue;
                var n = rend.gameObject.name;
                rend.enabled = n.StartsWith("cube");   // the pedestal is drawn by the stadium (identical box)
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
            // depth precision: the stadium is 300 m across and its decals sit 1 cm apart — a 0.05 m near plane z-fights
            var c = cam.GetComponent<Camera>();
            c.nearClipPlane = 0.2f;
            c.farClipPlane = 1000f;
            bc.target = Array.Find(physics.GetComponentsInChildren<MjBody>(true), b => b.name == "pelvis");
            bc.offset = new Vector3(4.4f, 0.9f, -2.2f); // front three-quarter: the athlete faces +x, the empty pedestals run along +z

            EditorSceneManager.SaveScene(scene, IronPedestalScene);
            var (p, rot) = binder.BindError();
            return $"{IronPedestalScene}: brain {brainFile}, bind error {p * 1000f:F2} mm / {rot:F3}°";
        }
    }
}
