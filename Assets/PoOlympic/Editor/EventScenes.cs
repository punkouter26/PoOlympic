using System;
using System.IO;
using System.Linq;
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

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
        /// extraYawDeg turns the athlete relative to the event direction (180 = facing away from the finish, event 9).
        /// </summary>
        public static GameObject PlaceStadium(Scene scene, int eventNum, int lane, float extraYawDeg = 0f)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(StadiumAsset) ?? throw new FileNotFoundException(StadiumAsset);
            var st = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            st.name = "Stadium";
            var (_, yaw) = VenueLane(eventNum, lane);
            yaw += extraYawDeg;
            st.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, yaw, 0f) * Quaternion.Euler(0f, 180f, 0f));
            var name = $"E{eventNum:00}_L{lane}";
            var anchor = Array.Find(st.GetComponentsInChildren<Transform>(true), t => t.name == name)
                         ?? throw new MissingReferenceException($"lane anchor {name} not in {StadiumAsset}");
            st.transform.position = -anchor.position;
            foreach (var c in st.GetComponentsInChildren<Component>(true))
                if (c is Collider || c is Rigidbody) throw new InvalidOperationException($"stadium must be render-only: {c.GetType().Name} on {c.name}");
            return st;
        }

        public const string IronPedestalHeatScene = "Assets/PoOlympic/Scenes/Event_IronPedestal_Heat.unity";
        public const string PedestalHeatSource = "training/assets/scene_pedestal8.xml";
        public const string PedestalHeatLayout = "training/assets/pedestal8_layout.json";
        // Phase Z: the official heat is a mixed meet — MATT in lanes 1/3/5/7, the zombie in 2/4/6/8
        // (training/tools/compose_mixed.py pedestal8 matt,zombie,…)
        public const string PedestalMixedSource = "training/assets/scene_pedestal8_mzmzmzmz.xml";
        public const string PedestalMixedLayout = "training/assets/pedestal8_mzmzmzmz_layout.json";
        public const string ZombieAsset = "Assets/PoOlympic/Art/Zombie.glb";
        public const string DefaultZombieRung0Brain = "zombie_rung0.onnx";

        /// <summary>Per-body assets: contract (Models/contract[_body].json), visual (glTF) and lane label letter.</summary>
        public static (string contract, string visual, string letter) BodyAssets(string body) => body switch
        {
            "matt" => ("Assets/PoOlympic/Models/contract.json", VisualBinding.MattAsset, "M"),
            "zombie" => ("Assets/PoOlympic/Models/contract_zombie.json", ZombieAsset, "Z"),
            _ => throw new ArgumentException($"unknown athlete body '{body}'"),
        };

        /// <summary>
        /// Event 1 official heat: 8 athletes on the 8 stadium pedestals (lane origins come from the venue layout, so
        /// physics pedestals = stadium pedestals), 16 pooled cubes, IronPedestalHeat + HUD. Mixed meet: every lane carries
        /// its own body (layout "body"), contract, brain and visual.
        /// </summary>
        [MenuItem("PoOlympic/Events/Build Event 1 — Iron Pedestal Heat (MATT + zombie)")]
        public static string BuildIronPedestalHeat() => BuildIronPedestalHeat(DefaultRung0Brain, PedestalMixedSource, PedestalMixedLayout);

        public static string BuildIronPedestalHeat(string brainFile) => BuildIronPedestalHeat(brainFile, PedestalHeatSource, PedestalHeatLayout);

        public static string BuildIronPedestalHeat(string brainFile, string source, string layoutPath, string zombieBrain = DefaultZombieRung0Brain)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            AthleteImport.SyncModel(layoutPath);
            var layout = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(Application.dataPath), layoutPath)));
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var physics = AthleteImport.ImportIntoActiveScene(source);

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
            PlaceStadium(scene, 1, AthleteLane);

            var cubeMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/PoOlympic/Materials/PoolCube.mat");
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
            {
                if (rend.GetComponent<MjGeom>() == null) continue;
                bool cube = rend.gameObject.name.StartsWith("cube");
                rend.enabled = cube;                         // pedestals are drawn by the stadium (identical boxes)
                if (cube && cubeMat != null) rend.sharedMaterial = cubeMat;
            }

            var brains = new System.Collections.Generic.Dictionary<string, string> { { "matt", brainFile }, { "zombie", zombieBrain } };
            var pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            pool.poolSize = (int)layout["n_cubes"];
            var heat = new GameObject("IronPedestalHeat").AddComponent<IronPedestalHeat>();
            heat.cubes = pool;
            MjBody focusPelvis = null;
            var lineup = new System.Collections.Generic.List<string>();
            foreach (var l in layout["lanes"])
            {
                int k = (int)l["lane"];
                string prefix = (string)l["prefix"];
                string body = (string)l["body"] ?? "matt";
                lineup.Add(body);
                var (contractPath, visualPath, letter) = BodyAssets(body);
                var o = l["origin"];
                var go = new GameObject($"Athlete_Lane{k + 1}");
                var r = go.AddComponent<PolicyRunner>();
                r.contractJson = AssetDatabase.LoadAssetAtPath<TextAsset>(contractPath) ?? throw new FileNotFoundException(contractPath);
                r.brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{brains[body]}") ?? throw new FileNotFoundException(brains[body]);
                r.brainSidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{brains[body]}.json");
                r.athletePrefix = prefix;
                r.laneOriginX = (double)o[0];
                r.laneOriginY = (double)o[1];
                r.cubeSlots = l["cubes"].Select(c => (int)c).ToArray();
                r.useStandardParityScript = false;
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(visualPath) ?? throw new FileNotFoundException(visualPath);
                var visual = (GameObject)PrefabUtility.InstantiatePrefab(prefab, go.transform);
                visual.name = $"{body.ToUpperInvariant()}_Visual";
                visual.transform.SetPositionAndRotation(new Vector3((float)o[0], (float)o[2], (float)o[1]), VisualBinding.GltfToPlugin);
                var binder = visual.AddComponent<BoneBinder>();
                binder.Capture(physics.transform, prefix);
                var (pe, re) = binder.BindError();
                if (pe > 0.01f || re > 1f) throw new InvalidOperationException($"lane {k} ({body}) bind error {pe * 1000f:F2} mm / {re:F2} deg");
                heat.runners.Add(new IronPedestalHeat.Runner { runner = r, name = $"{letter}{k + 1}" });
                if (k == AthleteLane) focusPelvis = Array.Find(physics.GetComponentsInChildren<MjBody>(true), bd => bd.name == prefix + "pelvis");
            }
            pool.runner = heat.runners[0].runner;
            var hud = new GameObject("HeatHUD").AddComponent<HeatHud>();
            hud.heat = heat;
            hud.version = lineup.Distinct().Count() > 1
                ? $"v0 · {Path.GetFileNameWithoutExtension(brainFile)} + {Path.GetFileNameWithoutExtension(zombieBrain)}"
                : $"v0 · {Path.GetFileNameWithoutExtension(brainFile)} · 8 runners";

            var cc = cam.GetComponent<Camera>();
            cc.nearClipPlane = 0.2f;
            cc.farClipPlane = 1000f;
            var bc = cam.GetComponent<BroadcastCamera>();
            bc.target = focusPelvis;
            bc.focusOffset = new Vector3(0f, 0f, 1.5f);      // centre of the row (lanes 1..8 at Unity z = -9 .. +12)
            bc.offset = new Vector3(8f, 3.4f, -17f);         // front-left end of the row: all 8 pedestals recede in a 9:16 frame
            EditorSceneManager.SaveScene(scene, IronPedestalHeatScene);
            return $"{IronPedestalHeatScene}: {heat.runners.Count} runners ({string.Join(",", lineup)}), brains {string.Join(" / ", brains.Values)}";
        }

        public const string TrackSource = "training/assets/scene_track8.xml";
        public const string TrackLayout = "training/assets/track8_layout.json";
        public const string DefaultRung2Brain = "rung2.onnx";
        public static readonly (int ev, TrackRaceEvent.Mode mode, string title, string scene)[] TrackEvents =
        {
            (8, TrackRaceEvent.Mode.Dash, "30m DASH", "Assets/PoOlympic/Scenes/Event_30mDash.unity"),
            (19, TrackRaceEvent.Mode.Terminal, "TERMINAL VELOCITY", "Assets/PoOlympic/Scenes/Event_TerminalVelocity.unity"),
            (22, TrackRaceEvent.Mode.Brake, "EMERGENCY BRAKE", "Assets/PoOlympic/Scenes/Event_EmergencyBrake.unity"),
            (9, TrackRaceEvent.Mode.Inverted, "INVERTED SPRINT", "Assets/PoOlympic/Scenes/Event_InvertedSprint.unity"),
        };

        [MenuItem("PoOlympic/Events/Build Track Races (8, 9, 19, 22)")]
        public static string BuildTrackRaces()
        {
            var sb = new System.Text.StringBuilder();
            foreach (var t in TrackEvents) sb.AppendLine(BuildTrackRace(t.ev, t.mode, t.title, t.scene, DefaultRung2Brain));
            return sb.ToString();
        }

        /// <summary>
        /// Straight-track race scene: 8 runners on scene_track8.xml (lane origins from the stadium venue), stadium snapped
        /// to the event's lane 4, TrackRaceEvent + RaceHud, broadcast camera trackside following lane 4.
        /// Backward races (Inverted): the athletes face away from the finish — the stadium is turned 180° about venue lane 5
        /// (the MuJoCo lane layout is the mirror image), so MuJoCo lane k runs in venue lane 8 - k.
        /// </summary>
        public static string BuildTrackRace(int eventNum, TrackRaceEvent.Mode mode, string title, string scenePath, string brainFile)
        {
            bool reversed = mode == TrackRaceEvent.Mode.Inverted;
            var meet = BuildMeetScene(TrackSource, TrackLayout, eventNum, reversed ? 7 - AthleteLane : AthleteLane, reversed ? 180f : 0f, brainFile);
            var race = new GameObject("TrackRaceEvent").AddComponent<TrackRaceEvent>();
            race.mode = mode;
            (race.distance, race.commandSpeed, race.maxSeconds) = TrackRaceEvent.Defaults(mode);
            foreach (var (k, r) in meet.Lanes)
                race.runners.Add(new TrackRaceEvent.Runner { runner = r, name = $"L{(reversed ? 8 - k : k + 1)}" });
            meet.Pool.runner = race.runners[0].runner;
            var hud = new GameObject("RaceHUD").AddComponent<RaceHud>();
            hud.race = race;
            hud.title = title;
            hud.subtitle = $"Event {eventNum} · 8 runners";
            hud.version = $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}";

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(reversed ? -1.0f : 1.0f, 0f, -0.6f);   // centre of the 8 lanes (Unity z = +3.66 .. -4.88), a bit ahead
            bc.offset = new Vector3(reversed ? 3.5f : -3.5f, 3.2f, -13f);      // trackside, slightly behind the pack (backward runners face it)
            EditorSceneManager.SaveScene(meet.Scene, scenePath);
            return $"{scenePath}: {mode}, {race.runners.Count} runners, {race.distance} m, brain {brainFile}";
        }

        public const string TurntableScene = "Assets/PoOlympic/Scenes/Event_360Turntable.unity";
        public const string TurntableSource = "training/assets/scene_turntable8.xml";
        public const string TurntableLayout = "training/assets/turntable8_layout.json";

        /// <summary>Event 12: 8 athletes on the venue's spin spots (scene_turntable8.xml), TurntableEvent + HUD.</summary>
        [MenuItem("PoOlympic/Events/Build Event 12 — 360 Turntable")]
        public static string BuildTurntable() => BuildTurntable(DefaultRung2Brain);

        public static string BuildTurntable(string brainFile)
        {
            var meet = BuildMeetScene(TurntableSource, TurntableLayout, 12, AthleteLane, 0f, brainFile);
            var ev = new GameObject("TurntableEvent").AddComponent<TurntableEvent>();
            foreach (var (k, r) in meet.Lanes) ev.spinners.Add(new TurntableEvent.Spinner { runner = r, name = $"S{k + 1}" });
            meet.Pool.runner = ev.spinners[0].runner;
            var hud = new GameObject("StandingsHUD").AddComponent<StandingsHud>();
            hud.board = ev;
            hud.title = "THE 360 TURNTABLE";
            hud.subtitle = "Event 12";
            hud.version = $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}";

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-4.5f, 0f, -2f);    // centre of the 2 x 4 grid (spot S4 is the origin)
            bc.offset = new Vector3(10.5f, 5.2f, -8.5f);     // front three-quarter: the athletes start facing +x
            EditorSceneManager.SaveScene(meet.Scene, TurntableScene);
            return $"{TurntableScene}: {ev.spinners.Count} athletes, {ev.turns} turns at {ev.spinRate} rad/s, brain {brainFile}";
        }

        public const string CrabScene = "Assets/PoOlympic/Scenes/Event_CrabShuffle.unity";
        public const string CrabSource = "training/assets/scene_crab8.xml";
        public const string CrabLayout = "training/assets/crab8_layout.json";

        /// <summary>Event 10: 8 athletes turned 90° to the course (they side-step to their right) between the physical
        /// rails of scene_crab8.xml; CrabShuffleEvent + StandingsHud.</summary>
        [MenuItem("PoOlympic/Events/Build Event 10 — Crab Shuffle")]
        public static string BuildCrabShuffle() => BuildCrabShuffle(DefaultRung2Brain);

        public static string BuildCrabShuffle(string brainFile)
        {
            var meet = BuildMeetScene(CrabSource, CrabLayout, 10, AthleteLane, 90f, brainFile);
            var layout = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(Application.dataPath), CrabLayout)));
            var ev = new GameObject("CrabShuffleEvent").AddComponent<CrabShuffleEvent>();
            ev.railGeoms = layout["props"].Select(pr => (string)pr["name"]).ToArray();
            foreach (var (k, r) in meet.Lanes) ev.racers.Add(new CrabShuffleEvent.Racer { runner = r, name = $"L{k + 1}" });
            meet.Pool.runner = ev.racers[0].runner;
            var hud = new GameObject("StandingsHUD").AddComponent<StandingsHud>();
            hud.board = ev;
            hud.title = "CRAB SHUFFLE";
            hud.subtitle = "Event 10";
            hud.version = $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}";

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-0.6f, 0f, -1.5f);  // centre of the 8 lanes (Unity x = +3.66 .. -4.88), a bit down the course
            bc.offset = new Vector3(9.5f, 4f, -5.5f);        // in front of lane 1, down the course: the row recedes, faces + rails
            EditorSceneManager.SaveScene(meet.Scene, CrabScene);
            return $"{CrabScene}: {ev.racers.Count} athletes, {ev.distance} m at {ev.sideSpeed} m/s, {ev.railGeoms.Length} rails, brain {brainFile}";
        }

        public const string SlalomScene = "Assets/PoOlympic/Scenes/Event_SlalomSprint.unity";
        public const string SlalomSource = "training/assets/scene_slalom8.xml";
        public const string SlalomLayout = "training/assets/slalom8_layout.json";

        /// <summary>Event 11: 8 runners weave through the 7 physical poles on each lane's centre line (scene_slalom8.xml);
        /// SlalomEvent + StandingsHud.</summary>
        [MenuItem("PoOlympic/Events/Build Event 11 — Slalom Sprint")]
        public static string BuildSlalom() => BuildSlalom(DefaultRung2Brain);

        public static string BuildSlalom(string brainFile)
        {
            var meet = BuildMeetScene(SlalomSource, SlalomLayout, 11, AthleteLane, 0f, brainFile);
            var ev = new GameObject("SlalomEvent").AddComponent<SlalomEvent>();
            foreach (var (k, r) in meet.Lanes) ev.racers.Add(new SlalomEvent.Racer { runner = r, name = $"L{k + 1}", lane = k });
            meet.Pool.runner = ev.racers[0].runner;
            var hud = new GameObject("StandingsHUD").AddComponent<StandingsHud>();
            hud.board = ev;
            hud.title = "SLALOM SPRINT";
            hud.subtitle = "Event 11";
            hud.version = $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}";

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(1.5f, 0f, -0.6f);   // centre of the 8 lanes (Unity z = +3.66 .. -4.88), ahead of the pack
            bc.offset = new Vector3(-4f, 3.4f, -12f);        // trackside, outside lane 8, slightly behind the pack
            EditorSceneManager.SaveScene(meet.Scene, SlalomScene);
            return $"{SlalomScene}: {ev.racers.Count} runners, {ev.nPoles} poles, {ev.speed} m/s, brain {brainFile}";
        }

        public const string GauntletScene = "Assets/PoOlympic/Scenes/Event_GustGauntlet.unity";
        public const string ShakerSource = "training/assets/scene_shaker8.xml";
        public const string ShakerLayout = "training/assets/shaker8_layout.json";

        /// <summary>Event 5: 8 athletes on spring-mounted shaker platforms (scene_shaker8.xml); GustGauntletEvent +
        /// StandingsHud. The platforms move, so they are drawn by their MuJoCo geoms (stadium hazard material) and the
        /// stadium's static shaker pads are hidden.</summary>
        [MenuItem("PoOlympic/Events/Build Event 5 — Gust Gauntlet")]
        public static string BuildGustGauntlet() => BuildGustGauntlet(DefaultRung2Brain);

        public static string BuildGustGauntlet(string brainFile)
        {
            var meet = BuildMeetScene(ShakerSource, ShakerLayout, 5, AthleteLane, 0f, brainFile);
            var pads = meet.Scene.GetRootGameObjects().First(g => g.name == "Stadium").GetComponentsInChildren<Renderer>(true)
                           .Where(r => r.name.StartsWith("E05_Shaker_")).ToArray();
            if (pads.Length != 8) throw new InvalidOperationException($"expected 8 stadium shaker pads, found {pads.Length}");
            var hazard = pads[0].sharedMaterial;
            foreach (var pad in pads) pad.enabled = false;
            int shown = 0;
            foreach (var g in UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (!g.name.EndsWith("_shaker") || !g.TryGetComponent<Renderer>(out var rend)) continue;
                rend.enabled = true;
                rend.sharedMaterial = hazard;
                shown++;
            }
            if (shown != 8) throw new InvalidOperationException($"expected 8 MuJoCo shaker platforms, found {shown}");
            var ev = new GameObject("GustGauntletEvent").AddComponent<GustGauntletEvent>();
            foreach (var (k, r) in meet.Lanes) ev.athletes.Add(new GustGauntletEvent.Athlete { runner = r, name = $"S{k + 1}" });
            meet.Pool.runner = ev.athletes[0].runner;
            var hud = new GameObject("StandingsHUD").AddComponent<StandingsHud>();
            hud.board = ev;
            hud.title = "THE GUST GAUNTLET";
            hud.subtitle = "Event 5";
            hud.version = $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}";

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-4.5f, 0f, -2f);    // centre of the 2 x 4 grid (platform S4 is the origin)
            bc.offset = new Vector3(10.5f, 5.2f, -8.5f);     // front three-quarter, as the turntable
            EditorSceneManager.SaveScene(meet.Scene, GauntletScene);
            return $"{GauntletScene}: {ev.athletes.Count} athletes, {ev.rounds} rounds, brain {brainFile}";
        }

        public sealed class MeetScene
        {
            public Scene Scene;
            public GameObject Camera;
            public MjCubePool Pool;
            public MjBody FocusPelvis;
            public readonly System.Collections.Generic.List<(int lane, PolicyRunner runner)> Lanes = new();
        }

        /// <summary>
        /// Shared setup of every 8-athlete event scene: the event MJCF (lane origins from the stadium venue), camera + sun
        /// from the testbed, the render-only stadium snapped to venue lane `anchorLane` (turned by extraYawDeg), one
        /// PolicyRunner + bound MATT visual per lane, the cube pool. MuJoCo geoms are hidden except the pool cubes (the
        /// stadium draws the props). Focus pelvis = lane AthleteLane (the MuJoCo origin).
        /// </summary>
        public static MeetScene BuildMeetScene(string source, string layoutPath, int eventNum, int anchorLane, float extraYawDeg, string brainFile)
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            ParityHarness.SyncArtifacts();
            AthleteImport.SyncModel(layoutPath);
            var layout = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(Application.dataPath), layoutPath)));
            var meet = new MeetScene { Scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single) };
            var physics = AthleteImport.ImportIntoActiveScene(source);
            var src = EditorSceneManager.OpenScene(ParityHarness.TestbedScene, OpenSceneMode.Additive);
            GameObject Copy(string name)
            {
                var clone = UnityEngine.Object.Instantiate(Array.Find(src.GetRootGameObjects(), g => g.name == name));
                clone.name = name;
                SceneManager.MoveGameObjectToScene(clone, meet.Scene);
                return clone;
            }
            meet.Camera = Copy("Main Camera");
            Copy("Sun");
            EditorSceneManager.CloseScene(src, true);
            SceneManager.SetActiveScene(meet.Scene);
            PlaceStadium(meet.Scene, eventNum, anchorLane, extraYawDeg);

            var cubeMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/PoOlympic/Materials/PoolCube.mat");
            foreach (var rend in physics.GetComponentsInChildren<Renderer>(true))
            {
                if (rend.GetComponent<MjGeom>() == null) continue;
                bool cube = rend.gameObject.name.StartsWith("cube");
                rend.enabled = cube;
                if (cube && cubeMat != null) rend.sharedMaterial = cubeMat;
            }
            var contract = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/PoOlympic/Models/contract.json");
            var brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}") ?? throw new FileNotFoundException(brainFile);
            var sidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{brainFile}.json");
            var mattPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(VisualBinding.MattAsset);
            meet.Pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            meet.Pool.poolSize = (int)layout["n_cubes"];
            foreach (var l in layout["lanes"])
            {
                int k = (int)l["lane"];
                string prefix = (string)l["prefix"];
                var o = l["origin"];
                var go = new GameObject($"Athlete_Lane{k + 1}");
                var r = go.AddComponent<PolicyRunner>();
                r.contractJson = contract;
                r.brain = brain;
                r.brainSidecar = sidecar;
                r.athletePrefix = prefix;
                r.laneOriginX = (double)o[0];
                r.laneOriginY = (double)o[1];
                r.cubeSlots = l["cubes"].Select(c => (int)c).ToArray();
                r.useStandardParityScript = false;
                var visual = (GameObject)PrefabUtility.InstantiatePrefab(mattPrefab, go.transform);
                visual.name = "MATT_Visual";
                visual.transform.SetPositionAndRotation(new Vector3((float)o[0], (float)o[2], (float)o[1]), VisualBinding.GltfToPlugin);
                var binder = visual.AddComponent<BoneBinder>();
                binder.Capture(physics.transform, prefix);
                var (pe, re) = binder.BindError();
                if (pe > 0.01f || re > 1f) throw new InvalidOperationException($"lane {k} bind error {pe * 1000f:F2} mm / {re:F2} deg");
                meet.Lanes.Add((k, r));
                if (k == AthleteLane) meet.FocusPelvis = Array.Find(physics.GetComponentsInChildren<MjBody>(true), bd => bd.name == prefix + "pelvis");
            }
            var cc = meet.Camera.GetComponent<Camera>();
            cc.nearClipPlane = 0.2f;
            cc.farClipPlane = 1000f;
            return meet;
        }

        public const string HubScene = "Assets/PoOlympic/Scenes/Stadium_Hub.unity";
        public const string CatalogJson = "Assets/PoOlympic/Art/Stadium/events_catalog.json";

        /// <summary>Event number → built event scene (grows as events are implemented).</summary>
        public static readonly System.Collections.Generic.Dictionary<int, string> EventScenePaths = new()
        {
            { 1, IronPedestalHeatScene },   // official 8-runner heat (solo practice: Event_IronPedestal.unity)
            { 5, GauntletScene },
            { 8, "Assets/PoOlympic/Scenes/Event_30mDash.unity" },
            { 9, "Assets/PoOlympic/Scenes/Event_InvertedSprint.unity" },
            { 10, CrabScene },
            { 11, SlalomScene },
            { 12, TurntableScene },
            { 19, "Assets/PoOlympic/Scenes/Event_TerminalVelocity.unity" },
            { 22, "Assets/PoOlympic/Scenes/Event_EmergencyBrake.unity" },
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

            int built = SetBuildScenes();
            return $"{HubScene}: {venues.Count} event venues, {venues.Count(x => x.Playable)} playable; build scenes {built}";
        }

        /// <summary>Build Settings: main menu (first = start scene, if built) → stadium hub → every built event scene.</summary>
        static int SetBuildScenes()
        {
            var list = new System.Collections.Generic.List<EditorBuildSettingsScene>();
            if (File.Exists(MainMenuScene)) list.Add(new EditorBuildSettingsScene(MainMenuScene, true));
            list.Add(new EditorBuildSettingsScene(HubScene, true));
            foreach (var path in EventScenePaths.Values) list.Add(new EditorBuildSettingsScene(path, true));
            list.Add(new EditorBuildSettingsScene(IronPedestalScene, true));
            EditorBuildSettings.scenes = list.ToArray();
            return list.Count;
        }

        public const string MainMenuScene = "Assets/PoOlympic/Scenes/MainMenu.unity";
        public const string MenuUxml = "Assets/PoOlympic/UI/MainMenu.uxml";
        public const string MenuTheme = "Assets/PoOlympic/UI/PoOlympicTheme.tss";
        public const string MenuPanelSettings = "Assets/PoOlympic/UI/PoOlympicPanelSettings.asset";

        /// <summary>
        /// Main menu: pick the athlete for each of the 8 lanes (only MATT so far) and one of the playable events
        /// (EventScenePaths, names/rules from the catalogue); PLAY loads the event scene. UI Toolkit (MainMenu.uxml),
        /// portrait reference 1080 × 1920. Becomes the first scene in Build Settings.
        /// </summary>
        [MenuItem("PoOlympic/Events/Build Main Menu")]
        public static string BuildMainMenu()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            var panel = AssetDatabase.LoadAssetAtPath<PanelSettings>(MenuPanelSettings);
            if (panel == null)
            {
                panel = ScriptableObject.CreateInstance<PanelSettings>();
                AssetDatabase.CreateAsset(panel, MenuPanelSettings);
            }
            panel.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(MenuTheme) ?? throw new FileNotFoundException(MenuTheme);
            panel.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            panel.referenceResolution = new Vector2Int(1080, 1920);
            panel.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight;
            panel.match = 1f;                                  // fit the portrait layout to the screen height
            EditorUtility.SetDirty(panel);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var cam = new GameObject("Main Camera") { tag = "MainCamera" }.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.047f, 0.071f, 0.125f);
            var go = new GameObject("MainMenu");
            var doc = go.AddComponent<UIDocument>();
            doc.panelSettings = panel;
            doc.visualTreeAsset = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(MenuUxml) ?? throw new FileNotFoundException(MenuUxml);
            var menu = go.AddComponent<MainMenuController>();
            var catalog = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(CatalogJson))["events"];
            foreach (var (num, path) in EventScenePaths.OrderBy(kv => kv.Key))
            {
                var e = catalog.First(x => (int)x["number"] == num);
                var lineup = Enumerable.Repeat("MATT", 8).ToArray();
                if (num == 1)
                {
                    var lay = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(Application.dataPath), PedestalMixedLayout)));
                    lineup = lay["lanes"].Select(l => ((string)l["body"] ?? "matt").ToUpperInvariant()).ToArray();
                }
                menu.events.Add(new MainMenuController.MenuEvent
                {
                    number = num, name = (string)e["name"], rules = (string)e["rules"],
                    brain = num == 1 ? "rung0 · MATT + ZOMBIE" : (string)e["brain"],
                    scene = Path.GetFileNameWithoutExtension(path), lineup = lineup,
                });
            }
            EditorSceneManager.SaveScene(scene, MainMenuScene);
            int n = SetBuildScenes();
            return $"{MainMenuScene}: {menu.events.Count} events, 8 lanes (roster: {string.Join(", ", MeetLineup.Roster)}); build scenes {n}";
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
