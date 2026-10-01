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
            StadiumLook.Dress(st);
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
        // Roster scenes (training/tools/compose_mixed.py <scene> roster): MATT (L<k>_) + zombie (Z<k>_) in every lane;
        // LaneLineup keeps the body picked in the main menu and switches the other off before MuJoCo compiles.
        public const string PedestalRosterSource = "training/assets/scene_pedestal8_roster.xml";
        public const string PedestalRosterLayout = "training/assets/pedestal8_roster_layout.json";
        public const string DefaultZombieRung2Brain = "zombie_rung2.onnx";
        public const string DefaultZombieRung0Brain = "zombie_rung0.onnx";

        /// <summary>Per-body assets: contract (Models/contract[_body].json), visual (glTF) and lane label letter.</summary>
        public static (string contract, string visual, string letter) BodyAssets(string body) => body switch
        {
            "matt" => ("Assets/PoOlympic/Models/contract.json", VisualBinding.MattAsset, "M"),
            "zombie" => ("Assets/PoOlympic/Models/contract_zombie.json", ZombieAsset, "Z"),
            "mattbio" => ("Assets/PoOlympic/Models/contract_mattbio.json", VisualBinding.MattAsset, "M"),   // MATT's skeleton
            _ => throw new ArgumentException($"unknown athlete body '{body}'"),
        };

        /// <summary>
        /// Event 1 official heat: 8 athletes shoulder to shoulder (0.7 m) on the stadium's iron beam (lane origins and the
        /// beam come from the venue layout, so physics beam = stadium beam; crowd scene: the athletes collide), 16 pooled
        /// cubes, IronPedestalHeat + HUD. Mixed meet: every lane carries its own body (layout "body"), contract, brain and
        /// visual.
        /// </summary>
        [MenuItem("PoOlympic/Events/Build Event 1 — Iron Pedestal Heat (MATT + zombie)")]
        public static string BuildIronPedestalHeat() => BuildIronPedestalHeat(DefaultRung0Brain, PedestalRosterSource, PedestalRosterLayout);

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
                rend.enabled = cube;                         // the beam is drawn by the stadium (identical box)
                if (cube && cubeMat != null) rend.sharedMaterial = cubeMat;
            }

            var brains = new System.Collections.Generic.Dictionary<string, string> { { "matt", brainFile }, { "zombie", zombieBrain } };
            var pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            pool.poolSize = (int)layout["n_cubes"];
            var heat = new GameObject("IronPedestalHeat").AddComponent<IronPedestalHeat>();
            heat.cubes = pool;
            MjBody focusPelvis = null;
            var lineup = new System.Collections.Generic.List<string>();
            var lanes = new GameObject("LaneLineup").AddComponent<LaneLineup>();
            lanes.physicsRoot = physics;
            lanes.pool = pool;
            lanes.focusLane = AthleteLane;
            foreach (var l in layout["lanes"])
            {
                int k = (int)l["lane"];
                string prefix = (string)l["prefix"];
                string body = (string)l["body"] ?? "matt";
                lineup.Add(body);
                var (contractPath, visualPath, letter) = BodyAssets(body);
                var o = l["origin"];
                var go = new GameObject($"Athlete_Lane{k + 1}_{body.ToUpperInvariant()}");
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
                var pelvis = Array.Find(physics.GetComponentsInChildren<MjBody>(true), bd => bd.name == prefix + "pelvis");
                lanes.entries.Add(new LaneLineup.Entry { lane = k, body = body, runner = r, pelvis = pelvis.gameObject });
                if (k == AthleteLane && body == "matt") focusPelvis = pelvis;
            }
            pool.runner = heat.runners[0].runner;
            string version = lineup.Distinct().Count() > 1
                ? $"v0 · {Path.GetFileNameWithoutExtension(brainFile)} + {Path.GetFileNameWithoutExtension(zombieBrain)}"
                : $"v0 · {Path.GetFileNameWithoutExtension(brainFile)} · 8 runners";
            lanes.broadcastCamera = cam.GetComponent<BroadcastCamera>();

            var cc = cam.GetComponent<Camera>();
            cc.nearClipPlane = 0.2f;
            cc.farClipPlane = 1000f;
            var bc = cam.GetComponent<BroadcastCamera>();
            bc.target = focusPelvis;
            bc.focusOffset = new Vector3(0f, 0f, 0.35f);     // centre of the row (lanes 1..8 at Unity z = -2.1 .. +2.8)
            bc.offset = new Vector3(4.6f, 2.4f, -6.4f);      // front-left end of the beam: the row recedes in a 9:16 frame
            AddBroadcast(heat, "IRON PEDESTAL", "Event 1 · last one standing", 1, version, cam, BroadcastDirector.Kind.Arena, Vector3.right);
            EditorSceneManager.SaveScene(scene, IronPedestalHeatScene);
            return $"{IronPedestalHeatScene}: {heat.runners.Count} athletes ({string.Join(",", lineup)}), brains {string.Join(" / ", brains.Values)}";
        }

        public const string TrackSource = "training/assets/scene_track8_roster.xml";
        public const string TrackLayout = "training/assets/track8_roster_layout.json";
        // Event 8 (30m All Fours): crawlers 1.1 m apart (venue 08; crowd scene) — the other track races keep 1.22 m lanes
        public const string CrawlSource = "training/assets/scene_crawl8_roster.xml";
        public const string CrawlLayout = "training/assets/crawl8_roster_layout.json";
        public const string DefaultRung2Brain = "rung2.onnx";
        // Event 8 (30m All Fours): crawl brains per body (training/poolympic/tasks/crawl_env.py)
        public const string AllFoursScene = "Assets/PoOlympic/Scenes/Event_30mAllFours.unity";
        public const string CrawlMattBrain = "crawl_matt.onnx";
        public const string CrawlZombieBrain = "crawl_zombie.onnx";
        // Event 13 (Steeplechase Jog): MATT's flight brain (tasks PoOlympic-Matt-Rung2-Flight, r2f_v3 it100); the zombie
        // keeps its Rung 2 brain
        public const string SteepleScene = "Assets/PoOlympic/Scenes/Event_SteeplechaseJog.unity";
        public const string FlightMattBrain = "r2f_v3_it100.onnx";
        public static readonly (int ev, TrackRaceEvent.Mode mode, string title, string scene)[] TrackEvents =
        {
            (8, TrackRaceEvent.Mode.AllFours, "30m ALL FOURS", AllFoursScene),   // replaced The 30m Dash (2026-09-29)
            (19, TrackRaceEvent.Mode.Terminal, "TERMINAL VELOCITY", "Assets/PoOlympic/Scenes/Event_TerminalVelocity.unity"),
            (22, TrackRaceEvent.Mode.Brake, "EMERGENCY BRAKE", "Assets/PoOlympic/Scenes/Event_EmergencyBrake.unity"),
            (9, TrackRaceEvent.Mode.Inverted, "INVERTED SPRINT", "Assets/PoOlympic/Scenes/Event_InvertedSprint.unity"),
            (13, TrackRaceEvent.Mode.Steeplechase, "STEEPLECHASE JOG", SteepleScene),
        };

        [MenuItem("PoOlympic/Events/Build Track Races (8, 9, 13, 19, 22)")]
        public static string BuildTrackRaces()
        {
            var sb = new System.Text.StringBuilder();
            foreach (var t in TrackEvents)
                sb.AppendLine(t.mode == TrackRaceEvent.Mode.AllFours
                    ? BuildTrackRace(t.ev, t.mode, t.title, t.scene, DefaultRung2Brain, CrawlSource, CrawlLayout)
                    : BuildTrackRace(t.ev, t.mode, t.title, t.scene, DefaultRung2Brain));
            return sb.ToString();
        }

        /// <summary>
        /// Straight-track race scene: 8 runners on scene_track8.xml (lane origins from the stadium venue), stadium snapped
        /// to the event's lane 4, TrackRaceEvent + RaceHud, broadcast camera trackside following lane 4.
        /// Backward races (Inverted): the athletes face away from the finish — the stadium is turned 180° about venue lane 5
        /// (the MuJoCo lane layout is the mirror image), so MuJoCo lane k runs in venue lane 8 - k.
        /// </summary>
        public static string BuildTrackRace(int eventNum, TrackRaceEvent.Mode mode, string title, string scenePath, string brainFile)
            => BuildTrackRace(eventNum, mode, title, scenePath, brainFile, TrackSource, TrackLayout);

        public static string BuildTrackRace(int eventNum, TrackRaceEvent.Mode mode, string title, string scenePath, string brainFile,
                                            string source, string layout, float distance = 0f)
        {
            bool reversed = mode == TrackRaceEvent.Mode.Inverted;
            bool crawl = mode == TrackRaceEvent.Mode.AllFours;
            bool steeple = mode == TrackRaceEvent.Mode.Steeplechase;
            var brains = crawl ? new System.Collections.Generic.Dictionary<string, string> { { "matt", CrawlMattBrain }, { "zombie", CrawlZombieBrain } }
                       : steeple ? new System.Collections.Generic.Dictionary<string, string> { { "matt", FlightMattBrain }, { "zombie", DefaultZombieRung2Brain } }
                       : null;
            if (crawl) brainFile = CrawlMattBrain;
            if (steeple) brainFile = FlightMattBrain;
            var meet = BuildMeetScene(source, layout, eventNum, reversed ? 7 - AthleteLane : AthleteLane, reversed ? 180f : 0f, brainFile, brains);
            var race = new GameObject("TrackRaceEvent").AddComponent<TrackRaceEvent>();
            race.mode = mode;
            (race.distance, race.commandSpeed, race.maxSeconds) = TrackRaceEvent.Defaults(mode);
            if (distance > 0f) race.distance = distance;
            foreach (var (k, r) in meet.Lanes)
            {
                race.runners.Add(new TrackRaceEvent.Runner { runner = r, name = LaneLabel($"L{(reversed ? 8 - k : k + 1)}", r) });
                if (crawl)
                {
                    // all_fours.py prone_start: face down, head towards the finish, pelvis at 0.22 m × λ (λ = SpeedScale²)
                    double s = Contract.Parse(r.contractJson.text).SpeedScale;
                    r.startProne = true;
                    r.proneHeight = 0.22 * s * s;
                    r.crawlSteering = true;
                    EditorUtility.SetDirty(r);
                }
            }
            meet.Pool.runner = race.runners[0].runner;

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(reversed ? -1.0f : 1.0f, 0f, -0.6f);   // centre of the 8 lanes (Unity z = +3.66 .. -4.88), a bit ahead
            bc.offset = new Vector3(reversed ? 3.5f : -3.5f, 3.2f, -13f);      // trackside, slightly behind the pack (backward runners face it)
            if (crawl) { bc.offset = new Vector3(-3.0f, 2.0f, -9f); bc.lookHeight = 0.4f; }   // crawlers are ~0.5 m tall
            AddBroadcast(race, title, $"Event {eventNum}", eventNum, BrainsLabel(brainFile), meet.Camera, BroadcastDirector.Kind.Race,
                         reversed ? Vector3.left : Vector3.right);
            EditorSceneManager.SaveScene(meet.Scene, scenePath);
            return $"{scenePath}: {mode}, {race.runners.Count} runners, {race.distance} m, brain {brainFile}";
        }

        public const string TrenchScene = "Assets/PoOlympic/Scenes/Event_TrenchCrawl.unity";
        public const string TrenchSource = "training/assets/scene_trench8_roster.xml";
        public const string TrenchLayout = "training/assets/trench8_roster_layout.json";
        public const float TrenchDistance = 16f;   // = all_fours.TRENCH_DISTANCE

        /// <summary>Event 23: the all-fours race (crawl brains) over the venue's 16 m under the trench ceiling
        /// (build_mjcf.trench_props: 12 m slab, underside 0.72 m, posts outside the lanes). The ceiling and posts are
        /// physical, so they are drawn by their MuJoCo geoms (translucent TrenchGlass slab, stadium steel posts); the
        /// stadium's own render-only trench (0.60 m, older export) is hidden.</summary>
        [MenuItem("PoOlympic/Events/Build Event 23 — The Trench Crawl")]
        public static string BuildTrenchCrawl()
        {
            var msg = BuildTrackRace(23, TrackRaceEvent.Mode.AllFours, "THE TRENCH CRAWL", TrenchScene, CrawlMattBrain, TrenchSource, TrenchLayout, TrenchDistance);
            var scene = EditorSceneManager.OpenScene(TrenchScene);
            var art = scene.GetRootGameObjects().First(g => g.name == "Stadium").GetComponentsInChildren<Renderer>(true)
                           .Where(r => r.name.StartsWith("E23_")).ToArray();
            var slab = TrenchGlass();                         // see-through: the broadcast cameras film the crawlers from above
            var steel = art.FirstOrDefault(r => r.name.Contains("Post"))?.sharedMaterial;
            int hidden = 0, shown = 0;
            foreach (var r in art.Where(r => r.name.Contains("Ceiling") || r.name.Contains("Post"))) { r.enabled = false; hidden++; }
            foreach (var g in UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (!g.name.StartsWith("trench_") || !g.TryGetComponent<Renderer>(out var rend)) continue;
                rend.enabled = true;
                rend.sharedMaterial = g.name.Contains("ceiling") ? slab : steel;
                shown++;
            }
            if (shown != 9) throw new InvalidOperationException($"expected 9 MuJoCo trench geoms (ceiling + 8 posts), found {shown}");

            EditorSceneManager.SaveScene(scene);
            return $"{msg}; trench: {shown} MuJoCo geoms drawn, {hidden} stadium renderers hidden";
        }

        public const string TrenchGlassMaterial = "Assets/PoOlympic/Materials/TrenchGlass.mat";

        /// <summary>Tinted, translucent URP material for the trench ceiling (an opaque slab hides the crawlers from every
        /// broadcast angle but a sub-0.7 m one, where it is an invisible 4 cm line).</summary>
        static Material TrenchGlass()
        {
            var mat = AssetDatabase.LoadAssetAtPath<Material>(TrenchGlassMaterial);
            if (mat != null) return mat;
            mat = new Material(Shader.Find("Universal Render Pipeline/Lit")) { name = "TrenchGlass" };
            mat.SetFloat("_Surface", 1f);                  // transparent
            mat.SetFloat("_Blend", 0f);                    // alpha
            mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetFloat("_ZWrite", 0f);
            mat.SetFloat("_Cull", 0f);                     // both faces (seen from above and below)
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            mat.SetColor("_BaseColor", new Color(0.55f, 0.75f, 0.95f, 0.28f));
            mat.SetFloat("_Smoothness", 0.85f);
            AssetDatabase.CreateAsset(mat, TrenchGlassMaterial);
            return mat;
        }

        public const string TurntableScene = "Assets/PoOlympic/Scenes/Event_360Turntable.unity";
        public const string TurntableSource = "training/assets/scene_turntable8_roster.xml";
        public const string TurntableLayout = "training/assets/turntable8_roster_layout.json";

        /// <summary>Event 12: 8 athletes on the venue's ring of spin spots, 0.75 m apart (scene_turntable8.xml, crowd scene),
        /// TurntableEvent + HUD.</summary>
        [MenuItem("PoOlympic/Events/Build Event 12 — 360 Turntable")]
        public static string BuildTurntable() => BuildTurntable(DefaultRung2Brain);

        public static string BuildTurntable(string brainFile)
        {
            var meet = BuildMeetScene(TurntableSource, TurntableLayout, 12, AthleteLane, 0f, brainFile);
            var ev = new GameObject("TurntableEvent").AddComponent<TurntableEvent>();
            foreach (var (k, r) in meet.Lanes) ev.spinners.Add(new TurntableEvent.Spinner { runner = r, name = LaneLabel($"S{k + 1}", r) });
            meet.Pool.runner = ev.spinners[0].runner;

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-0.69f, 0f, 0.69f); // centre of the ring (spot S4 is the origin)
            bc.offset = new Vector3(5.2f, 3.6f, -5.2f);      // front three-quarter: the athletes start facing +x
            AddBroadcast(ev, "THE 360 TURNTABLE", "Event 12", 12, BrainsLabel(brainFile), meet.Camera, BroadcastDirector.Kind.Arena, Vector3.right);
            EditorSceneManager.SaveScene(meet.Scene, TurntableScene);
            return $"{TurntableScene}: {ev.spinners.Count} athletes, {ev.turns} turns at {ev.spinRate} rad/s, brain {brainFile}";
        }

        public const string SquatScene = "Assets/PoOlympic/Scenes/Event_DeepSquat.unity";
        public const string SquatSource = "training/assets/scene_squat8_mattbio.xml";
        public const string SquatLayout = "training/assets/squat8_mattbio_layout.json";
        // Event 3: the Rung S brain (contract v4 pelvis-height command), trained on the mattbio body -> all-mattbio scene
        public const string SquatBrain = "rs_v6_it3299.onnx";   // squat 10/10, worst depth error 1.5 cm

        /// <summary>Event 3: 8 mattbio athletes on the venue's 2 x 4 station grid (3 m x 4 m apart, scene_squat8_mattbio.xml),
        /// DeepSquatEvent + HUD.</summary>
        [MenuItem("PoOlympic/Events/Build Event 3 — Deep Squat Endurance")]
        public static string BuildDeepSquat() => BuildDeepSquat(SquatBrain);

        public static string BuildDeepSquat(string brainFile)
        {
            var meet = BuildMeetScene(SquatSource, SquatLayout, 3, AthleteLane, 0f, brainFile,
                                      new System.Collections.Generic.Dictionary<string, string> { { "mattbio", brainFile } });
            var ev = new GameObject("DeepSquatEvent").AddComponent<DeepSquatEvent>();
            foreach (var (k, r) in meet.Lanes) ev.squatters.Add(new DeepSquatEvent.Squatter { runner = r, name = $"S{k + 1}" });
            meet.Pool.runner = ev.squatters[0].runner;

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-4.5f, 0f, -2f);    // grid centre (station S4 is the origin; rows at y = 0 / -4 m)
            bc.offset = new Vector3(8.5f, 4.0f, -5.0f);      // front three-quarter: the athletes face +x
            AddBroadcast(ev, "DEEP SQUAT", "Event 3", 3, $"v0 · {Path.GetFileNameWithoutExtension(brainFile)}", meet.Camera,
                         BroadcastDirector.Kind.Arena, Vector3.right);
            EditorSceneManager.SaveScene(meet.Scene, SquatScene);
            return $"{SquatScene}: {ev.squatters.Count} athletes, {ev.reps} reps, brain {brainFile}";
        }

        public const string CrabScene = "Assets/PoOlympic/Scenes/Event_CrabShuffle.unity";
        public const string CrabSource = "training/assets/scene_crab8_roster.xml";
        public const string CrabLayout = "training/assets/crab8_roster_layout.json";

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
            foreach (var (k, r) in meet.Lanes) ev.racers.Add(new CrabShuffleEvent.Racer { runner = r, name = LaneLabel($"L{k + 1}", r) });
            meet.Pool.runner = ev.racers[0].runner;

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-0.6f, 0f, -1.5f);  // centre of the 8 lanes (Unity x = +3.66 .. -4.88), a bit down the course
            bc.offset = new Vector3(9.5f, 4f, -5.5f);        // in front of lane 1, down the course: the row recedes, faces + rails
            AddBroadcast(ev, "CRAB SHUFFLE", "Event 10", 10, BrainsLabel(brainFile), meet.Camera, BroadcastDirector.Kind.Race, Vector3.right);
            EditorSceneManager.SaveScene(meet.Scene, CrabScene);
            return $"{CrabScene}: {ev.racers.Count} athletes, {ev.distance} m at {ev.sideSpeed} m/s, {ev.railGeoms.Length} rails, brain {brainFile}";
        }

        public const string SlalomScene = "Assets/PoOlympic/Scenes/Event_SlalomSprint.unity";
        public const string SlalomSource = "training/assets/scene_slalom8_roster.xml";
        public const string SlalomLayout = "training/assets/slalom8_roster_layout.json";

        /// <summary>Event 11: 8 runners weave through the 7 physical poles on each lane's centre line (scene_slalom8.xml,
        /// 1.4 m lanes, crowd scene: mirror slalom — neighbours meet at every other pole); SlalomEvent + StandingsHud.</summary>
        [MenuItem("PoOlympic/Events/Build Event 11 — Slalom Sprint")]
        public static string BuildSlalom() => BuildSlalom(DefaultRung2Brain);

        public static string BuildSlalom(string brainFile)
        {
            var meet = BuildMeetScene(SlalomSource, SlalomLayout, 11, AthleteLane, 0f, brainFile);
            var ev = new GameObject("SlalomEvent").AddComponent<SlalomEvent>();
            foreach (var (k, r) in meet.Lanes) ev.racers.Add(new SlalomEvent.Racer { runner = r, name = LaneLabel($"L{k + 1}", r), lane = k });
            meet.Pool.runner = ev.racers[0].runner;

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(1.5f, 0f, -0.7f);   // centre of the 8 lanes (Unity z = +4.2 .. -5.6), ahead of the pack
            bc.offset = new Vector3(-4f, 3.4f, -12f);        // trackside, outside lane 8, slightly behind the pack
            AddBroadcast(ev, "SLALOM SPRINT", "Event 11", 11, BrainsLabel(brainFile), meet.Camera, BroadcastDirector.Kind.Race, Vector3.right);
            EditorSceneManager.SaveScene(meet.Scene, SlalomScene);
            return $"{SlalomScene}: {ev.racers.Count} runners, {ev.nPoles} poles, {ev.speed} m/s, brain {brainFile}";
        }

        public const string GauntletScene = "Assets/PoOlympic/Scenes/Event_GustGauntlet.unity";
        public const string ShakerSource = "training/assets/scene_shaker8_roster.xml";
        public const string ShakerLayout = "training/assets/shaker8_roster_layout.json";

        /// <summary>Event 5: 8 athletes in a 2 x 4 grid (0.8 m) on ONE spring-mounted shaker floor (scene_shaker8.xml, crowd
        /// scene); GustGauntletEvent + StandingsHud. The floor moves, so it is drawn by its MuJoCo geom (stadium hazard
        /// material) and the stadium's static floor is hidden.</summary>
        [MenuItem("PoOlympic/Events/Build Event 5 — Gust Gauntlet")]
        public static string BuildGustGauntlet() => BuildGustGauntlet(DefaultRung2Brain);

        public static string BuildGustGauntlet(string brainFile)
        {
            var meet = BuildMeetScene(ShakerSource, ShakerLayout, 5, AthleteLane, 0f, brainFile);
            var pads = meet.Scene.GetRootGameObjects().First(g => g.name == "Stadium").GetComponentsInChildren<Renderer>(true)
                           .Where(r => r.name.StartsWith("E05_Shaker")).ToArray();
            if (pads.Length != 1) throw new InvalidOperationException($"expected the stadium shaker floor, found {pads.Length}");
            var hazard = pads[0].sharedMaterial;
            foreach (var pad in pads) pad.enabled = false;
            int shown = 0;
            foreach (var g in UnityEngine.Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (!(g.name == "shaker" || g.name.EndsWith("_shaker")) || !g.TryGetComponent<Renderer>(out var rend)) continue;
                rend.enabled = true;
                rend.sharedMaterial = hazard;
                shown++;
            }
            if (shown != 1) throw new InvalidOperationException($"expected 1 MuJoCo shaker floor, found {shown}");   // shared by all lanes
            var ev = new GameObject("GustGauntletEvent").AddComponent<GustGauntletEvent>();
            foreach (var (k, r) in meet.Lanes) ev.athletes.Add(new GustGauntletEvent.Athlete { runner = r, name = LaneLabel($"S{k + 1}", r), lane = k });
            meet.Pool.runner = ev.athletes[0].runner;

            var bc = meet.Camera.GetComponent<BroadcastCamera>();
            bc.target = meet.FocusPelvis;
            bc.focusOffset = new Vector3(-1.2f, 0f, -0.4f);  // centre of the 2 x 4 grid (spot S4 is the origin)
            bc.offset = new Vector3(5.6f, 3.8f, -5.6f);      // front three-quarter, as the turntable
            AddBroadcast(ev, "THE GUST GAUNTLET", "Event 5", 5, BrainsLabel(brainFile), meet.Camera, BroadcastDirector.Kind.Arena, Vector3.right);
            EditorSceneManager.SaveScene(meet.Scene, GauntletScene);
            return $"{GauntletScene}: {ev.athletes.Count} athletes, {ev.rounds} rounds, brain {brainFile}";
        }

        public const string HudStyle = "Assets/PoOlympic/UI/BroadcastHud.uss";
        public const string OddsModel = "Assets/PoOlympic/Models/odds_model.json";

        /// <summary>D3 broadcast layer of an 8-athlete event scene: UI Toolkit BroadcastHud (standings + odds, betting slip,
        /// ticker, result card, records, gauntlet) and a BroadcastDirector on the camera (tracking shots; the authored
        /// BroadcastCamera offset becomes the establishing shot).</summary>
        static BroadcastHud AddBroadcast(MonoBehaviour board, string title, string subtitle, int eventNum, string version, GameObject cam,
                                         BroadcastDirector.Kind kind, Vector3 forward)
        {
            var go = new GameObject("BroadcastHUD");
            var doc = go.AddComponent<UIDocument>();
            doc.panelSettings = AssetDatabase.LoadAssetAtPath<PanelSettings>(MenuPanelSettings) ?? throw new FileNotFoundException(MenuPanelSettings);
            var hud = go.AddComponent<BroadcastHud>();
            hud.board = board;
            hud.title = title;
            hud.subtitle = subtitle;
            hud.eventNumber = eventNum;
            hud.version = version;
            hud.style = AssetDatabase.LoadAssetAtPath<StyleSheet>(HudStyle) ?? throw new FileNotFoundException(HudStyle);
            hud.oddsModel = AssetDatabase.LoadAssetAtPath<TextAsset>(OddsModel);
            var bc = cam.GetComponent<BroadcastCamera>();
            var dir = cam.GetComponent<BroadcastDirector>() ?? cam.AddComponent<BroadcastDirector>();
            dir.board = board;
            dir.kind = kind;
            dir.forward = forward;
            dir.establishing = bc.offset;
            dir.lookHeight = bc.lookHeight;
            // features 2/3/5/6/7/10: tension meter, Cinemachine rig, impact FX, audio, per-athlete telemetry + overlays
            BroadcastFx.Install(hud, dir, UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include));
            return hud;
        }

        /// <summary>HUD lane label: the zombie's lanes read Z&lt;n&gt; (roster prefix Z&lt;k&gt;_), MATT keeps the event's own label.</summary>
        static string LaneLabel(string label, PolicyRunner r) => r.athletePrefix.StartsWith("Z") ? "Z" + label.Substring(1) : label;

        static string BrainsLabel(string brainFile) =>
            $"v0 · {Path.GetFileNameWithoutExtension(brainFile)} + " +
            Path.GetFileNameWithoutExtension(brainFile == CrawlMattBrain ? CrawlZombieBrain : DefaultZombieRung2Brain);

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
        public static MeetScene BuildMeetScene(string source, string layoutPath, int eventNum, int anchorLane, float extraYawDeg, string brainFile,
                                               System.Collections.Generic.Dictionary<string, string> bodyBrains = null)
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
            // per-lane body (layout "body", tools/compose_mixed.py; default matt): its contract, brain and visual
            bodyBrains ??= new System.Collections.Generic.Dictionary<string, string> { { "matt", brainFile }, { "zombie", DefaultZombieRung2Brain } };
            string BrainOf(string body) => bodyBrains.TryGetValue(body, out var b) ? b : brainFile;
            meet.Pool = new GameObject("CubePool").AddComponent<MjCubePool>();
            meet.Pool.poolSize = (int)layout["n_cubes"];
            var lanes = new GameObject("LaneLineup").AddComponent<LaneLineup>();
            lanes.physicsRoot = physics;
            lanes.pool = meet.Pool;
            lanes.focusLane = AthleteLane;
            lanes.broadcastCamera = meet.Camera.GetComponent<BroadcastCamera>();
            foreach (var l in layout["lanes"])
            {
                int k = (int)l["lane"];
                string prefix = (string)l["prefix"];
                string body = (string)l["body"] ?? "matt";
                var (contractPath, visualPath, _) = BodyAssets(body);
                var o = l["origin"];
                var go = new GameObject($"Athlete_Lane{k + 1}_{body.ToUpperInvariant()}");
                var r = go.AddComponent<PolicyRunner>();
                r.contractJson = AssetDatabase.LoadAssetAtPath<TextAsset>(contractPath) ?? throw new FileNotFoundException(contractPath);
                r.brain = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ParityHarness.ModelsFolder}/Brains/{BrainOf(body)}") ?? throw new FileNotFoundException(BrainOf(body));
                r.brainSidecar = AssetDatabase.LoadAssetAtPath<TextAsset>($"{ParityHarness.ModelsFolder}/Brains/{BrainOf(body)}.json");
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
                if (pe > 0.01f || re > 1f) throw new InvalidOperationException($"lane {k} bind error {pe * 1000f:F2} mm / {re:F2} deg");
                meet.Lanes.Add((k, r));
                var pelvis = Array.Find(physics.GetComponentsInChildren<MjBody>(true), bd => bd.name == prefix + "pelvis");
                lanes.entries.Add(new LaneLineup.Entry { lane = k, body = body, runner = r, pelvis = pelvis.gameObject });
                if (k == AthleteLane && (body == "matt" || meet.FocusPelvis == null)) meet.FocusPelvis = pelvis;   // MATT, else the lane's body
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
            { 3, SquatScene },
            { 5, GauntletScene },
            { 8, AllFoursScene },
            { 9, "Assets/PoOlympic/Scenes/Event_InvertedSprint.unity" },
            { 10, CrabScene },
            { 11, SlalomScene },
            { 12, TurntableScene },
            { 13, SteepleScene },
            { 19, "Assets/PoOlympic/Scenes/Event_TerminalVelocity.unity" },
            { 22, "Assets/PoOlympic/Scenes/Event_EmergencyBrake.unity" },
            { 23, TrenchScene },
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
            StadiumLook.Dress(st);

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
                menu.events.Add(new MainMenuController.MenuEvent   // roster scenes: any lineup (LaneLineup)
                {
                    number = num, name = (string)e["name"], rules = (string)e["rules"],
                    brain = num == 1 ? "rung0" : (string)e["brain"],
                    scene = Path.GetFileNameWithoutExtension(path), lineup = null,
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
