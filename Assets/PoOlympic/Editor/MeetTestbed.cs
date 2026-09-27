using System;
using System.Collections.Generic;
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
    /// C6 — the 8-lane testbed (DESIGN.md §4, gate G6). Builds Testbed_Rung1.unity from the generated meet MJCF
    /// (training/assets/scene_meet8.xml + meet8_layout.json): one PolicyRunner + bound MATT visual per lane, one shared
    /// 16-cube pool, and the camera / light / track / HUD look of Testbed_ZeroBrain. ConfigureG6 loads a lane plan written
    /// by training/tools/make_g6.py (commands, lane-local disturbance scripts, recording names).
    /// </summary>
    public static class MeetTestbed
    {
        public const string ScenePath = "Assets/PoOlympic/Scenes/Testbed_Rung1.unity";
        const string ContractAsset = "Assets/PoOlympic/Models/contract.json";

        [Serializable] class LaneLayout { public int lane; public string prefix; public double[] origin; public int[] cubes; }
        [Serializable] class Layout { public int n_lanes; public double lane_width; public int n_cubes; public LaneLayout[] lanes; }
        [Serializable] public class PlanLane { public int lane; public double[] command; public Disturbance[] disturbances; public string reference; }
        [Serializable] public class Plan { public string brain; public string onnx_sha256; public string fingerprint_sha256; public double seconds; public PlanLane[] lanes; }

        static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

        static Layout LoadLayout() =>
            JsonUtility.FromJson<Layout>(File.ReadAllText(Path.Combine(ProjectRoot, AthleteImport.MeetLayoutRelative)));

        [MenuItem("PoOlympic/Build Testbed_Rung1 (8 lanes)")]
        public static string Build()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            var layout = LoadLayout();
            AthleteImport.SyncModel(AthleteImport.MeetLayoutRelative);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var physics = AthleteImport.ImportIntoActiveScene(AthleteImport.MeetSourceRelative);

            // Look & feel copied from the zero-brain testbed (camera, light, render-only track, HUD).
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

            // Physics geoms are simulation-only; cubes stay visible with the pool material.
            var cubeMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/PoOlympic/Materials/PoolCube.mat");
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
            {
                if (rend.GetComponent<MjGeom>() == null) continue;
                bool cube = rend.gameObject.name.StartsWith("cube");
                rend.enabled = cube;
                if (cube && cubeMat != null) rend.sharedMaterial = cubeMat;
            }

            var contract = AssetDatabase.LoadAssetAtPath<TextAsset>(ContractAsset);
            var mattPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(VisualBinding.MattAsset);
            var runners = new List<PolicyRunner>();
            var pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            pool.poolSize = layout.n_cubes;
            foreach (var l in layout.lanes)
            {
                var go = new GameObject($"Athlete_Lane{l.lane}");
                var r = go.AddComponent<PolicyRunner>();
                r.contractJson = contract;
                r.athletePrefix = l.prefix;
                r.laneOriginX = l.origin[0];
                r.laneOriginY = l.origin[1];
                r.cubeSlots = l.cubes;
                r.holdDefaultPose = true; // until a brain is assigned (ConfigureG6 / SetBrain)
                r.useStandardParityScript = false;
                runners.Add(r);

                var visual = (GameObject)PrefabUtility.InstantiatePrefab(mattPrefab, go.transform);
                visual.name = "MATT_Visual";
                // MuJoCo (x, y, z) sits on Unity (x, z, y) in the plug-in; the bind pose is captured in place.
                visual.transform.SetPositionAndRotation(new Vector3((float)l.origin[0], 0f, (float)l.origin[1]), VisualBinding.GltfToPlugin);
                var binder = visual.AddComponent<BoneBinder>();
                binder.Capture(physics.transform, l.prefix);
                var (p, rot) = binder.BindError();
                if (p > 0.01f || rot > 1f) throw new InvalidOperationException($"lane {l.lane} bind error {p * 1000f:F2} mm / {rot:F2}°");
            }
            pool.runner = runners[0];

            var mid = runners[layout.n_lanes / 2];
            var bc = cam.GetComponent<BroadcastCamera>();
            bc.target = Array.Find(physics.GetComponentsInChildren<MjBody>(true), b => b.name == mid.athletePrefix + "pelvis");
            bc.offset = new Vector3(-2.5f, 2.6f, -11f); // right-hand stand, slightly behind: all 8 lanes in a 9:16 frame
            var hud = hudGo.GetComponent<TestbedHud>();
            hud.runner = mid;
            hud.lanes = runners.ToArray();
            hud.cubes = pool;
            hud.title = "PoOlympics — Testbed · 8 lanes";
            hud.version = "v0 · rung 1 · G6";

            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"[MeetTestbed] built {ScenePath}: {runners.Count} lanes, {layout.n_cubes} pool cubes");
            return ScenePath;
        }

        /// <summary>Assign one brain to every lane (holdDefaultPose off).</summary>
        public static void SetBrain(string brainFile)
        {
            var brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}")
                        ?? throw new FileNotFoundException($"brain {brainFile} not synced — PoOlympic › Parity › Sync Contract + Brains");
            var sidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}.json");
            foreach (var r in UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsSortMode.None))
            {
                r.brain = brain;
                r.brainSidecar = sidecar;
                r.holdDefaultPose = false;
                EditorUtility.SetDirty(r);
            }
        }

        /// <summary>Open Testbed_Rung1 and load parity/g6_plan_&lt;name&gt;.json into the lanes (records unity_run_g6_&lt;name&gt;_L&lt;k&gt;).</summary>
        public static string ConfigureG6(string name)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var plan = JsonUtility.FromJson<Plan>(File.ReadAllText(Path.Combine(ProjectRoot, "parity", $"g6_plan_{name}.json")));
            var contract = ParityHarness.LoadContract();
            if (plan.fingerprint_sha256 != contract.fingerprint_sha256)
                throw new InvalidOperationException("G6 plan was made on a different model than the contract");
            SetBrain(plan.brain);
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsSortMode.None);
            int ticks = (int)Math.Round(plan.seconds / (contract.timestep * contract.decimation));
            foreach (var pl in plan.lanes)
            {
                var r = Array.Find(runners, x => x.athletePrefix == $"L{pl.lane}_") ?? throw new KeyNotFoundException($"lane {pl.lane}");
                r.command = new Vector3((float)pl.command[0], (float)pl.command[1], (float)pl.command[2]);
                if (r.command.x != pl.command[0] || r.command.z != pl.command[2]) throw new InvalidOperationException($"lane {pl.lane}: command not float32-exact");
                r.useStandardParityScript = false;
                r.disturbances = new List<Disturbance>(pl.disturbances);
                r.recordName = $"g6_{name}_L{pl.lane}";
                r.recordTicks = ticks;
                EditorUtility.SetDirty(r);
            }
            EditorSceneManager.SaveOpenScenes();
            return $"G6 plan {name}: {plan.lanes.Length} lanes, brain {plan.brain}, {ticks} ticks";
        }
    }
}
