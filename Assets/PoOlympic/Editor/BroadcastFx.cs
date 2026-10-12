using System;
using System.IO;
using System.Linq;
using Unity.Cinemachine;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.VFX;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Broadcast FX layer of an 8-athlete event scene (features 2, 3, 5, 6, 7, 10), authored into the scene so every
    /// static object stays editable in the editor (nothing is spawned at runtime):
    ///   BroadcastFX/TensionMeter      shared excitement / danger model
    ///   BroadcastFX/Cinemachine       focus / hot / winner proxies + 5 CinemachineCameras (establishing-trackside,
    ///                                 head-on, high-wide, hot close-up, winner) with impulse listeners; CinemachineBrain
    ///                                 on the main camera; BroadcastDirector switches between them
    ///   BroadcastFX/ImpactFx          pooled dust / slam / shockwave particle systems, VFX Graph sparks (+ Shuriken
    ///                                 fallback), impulse source, hit-stop
    ///   BroadcastFX/ArenaAudio        crowd bed + tension layer + stingers + UI + 3D voice pool, clips from Audio/
    ///   per athlete                   AthleteTelemetry, the brain's critic (&lt;brain&gt;.critic.onnx)
    /// Idempotent: re-running replaces the previous layer. Called by EventScenes.AddBroadcast for new builds and by the
    /// menu for the existing event scenes. Everything is render / audio only — no Collider / Rigidbody (PhysXGuard).
    /// </summary>
    public static class BroadcastFx
    {
        public const string FxFolder = "Assets/PoOlympic/Art/Fx";
        public const string FxPrefabs = "Assets/PoOlympic/Prefabs/Fx";
        public const string AudioFolder = "Assets/PoOlympic/Audio";
        public const string ConfidenceModel = "Assets/PoOlympic/Models/confidence_model.json";
        public const string SparksVfx = "Assets/Samples/Visual Effect Graph/17.6.0/Visual Effect Graph Additions/VFX/Sparks.vfx";
        const string NoiseProfile = "Packages/com.unity.cinemachine/Presets/Noise/Handheld_normal_mild.asset";
        const string RootName = "BroadcastFX";

        // ------------------------------------------------------------------------------------------------ menu
        [MenuItem("PoOlympic/Broadcast/Upgrade broadcast FX (all event scenes)")]
        public static string UpgradeAll()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            EnsureAssets();
            var sb = new System.Text.StringBuilder();
            foreach (var path in EventScenes.EventScenePaths.Values.Distinct())
            {
                var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                sb.AppendLine($"{path}: {InstallInOpenScene()}");
                EditorSceneManager.SaveScene(scene);
            }
            return sb.ToString();
        }

        [MenuItem("PoOlympic/Broadcast/Upgrade broadcast FX (open scene)")]
        public static string UpgradeOpenScene()
        {
            EnsureAssets();
            var r = InstallInOpenScene();
            EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
            return r;
        }

        /// <summary>Install into the active scene (its BroadcastHud + BroadcastDirector + PolicyRunners).</summary>
        public static string InstallInOpenScene()
        {
            var hud = UnityEngine.Object.FindAnyObjectByType<BroadcastHud>(FindObjectsInactive.Include);
            var director = UnityEngine.Object.FindAnyObjectByType<BroadcastDirector>(FindObjectsInactive.Include);
            if (hud == null || director == null) return "skipped (no BroadcastHud / BroadcastDirector)";
            var runners = UnityEngine.Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include);
            Install(hud, director, runners);
            return $"installed ({runners.Length} athletes, {director.kind})";
        }

        // ------------------------------------------------------------------------------------------------ install
        public static void Install(BroadcastHud hud, BroadcastDirector director, PolicyRunner[] runners)
        {
            EnsureAssets();
            var scene = director.gameObject.scene;
            foreach (var old in scene.GetRootGameObjects().Where(g => g.name == RootName)) UnityEngine.Object.DestroyImmediate(old);
            var root = new GameObject(RootName);
            UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(root, scene);

            // tension
            var tension = Child(root, "TensionMeter").AddComponent<TensionMeter>();
            tension.board = hud.board;
            tension.kind = director.kind;
            tension.forward = director.forward;

            // cinemachine rig
            var cam = director.GetComponent<Camera>();
            var brain = director.GetComponent<CinemachineBrain>() ?? director.gameObject.AddComponent<CinemachineBrain>();
            brain.DefaultBlend = new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.Cut, 0f);
            brain.UpdateMethod = CinemachineBrain.UpdateMethods.LateUpdate;
            brain.IgnoreTimeScale = false;           // hit-stop freezes the camera too
            if (director.GetComponent<AudioListener>() == null) director.gameObject.AddComponent<AudioListener>();
            var rig = Child(root, "Cinemachine");
            var focus = Child(rig, "FocusProxy").transform;
            var hot = Child(rig, "HotProxy").transform;
            var winner = Child(rig, "WinnerProxy").transform;
            var leaderT = Child(rig, "LeaderProxy").transform;
            var noise = AssetDatabase.LoadAssetAtPath<NoiseSettings>(NoiseProfile);
            var lens = LensSettings.FromCamera(cam);
            CinemachineCamera Cam(string name, Transform target, float damping, bool handheld, int priority, float screenY = 0f)
            {
                var go = Child(rig, name);
                go.transform.position = cam.transform.position;
                go.transform.rotation = cam.transform.rotation;
                var c = go.AddComponent<CinemachineCamera>();
                c.Lens = lens;
                c.Follow = target;
                c.LookAt = target;
                c.Priority = priority;
                c.StandbyUpdate = CinemachineVirtualCameraBase.StandbyUpdateMode.Always;
                var follow = go.AddComponent<CinemachineFollow>();
                follow.TrackerSettings.BindingMode = Unity.Cinemachine.TargetTracking.BindingMode.WorldSpace;
                follow.TrackerSettings.PositionDamping = Vector3.one * damping;
                var aim = go.AddComponent<CinemachineRotationComposer>();
                aim.Damping = new Vector2(damping * 0.5f, damping * 0.5f);
                // the HUD docks above / below the viewport (BroadcastHud), so nothing overlaps: centred framing
                var comp = aim.Composition;
                comp.ScreenPosition = new Vector2(0f, screenY);
                aim.Composition = comp;
                if (handheld && noise != null)
                {
                    var perlin = go.AddComponent<CinemachineBasicMultiChannelPerlin>();
                    perlin.NoiseProfile = noise;
                    perlin.AmplitudeGain = 0.6f;
                    perlin.FrequencyGain = 0.8f;
                }
                var listener = go.AddComponent<CinemachineImpulseListener>();
                listener.Gain = 1f;
                listener.Use2DDistance = true;
                return c;
            }
            director.wide = Cam("CM_Establishing_Trackside", focus, 0.4f, false, 20);
            director.headOn = Cam("CM_HeadOn", focus, 0.3f, true, 10);
            director.high = Cam("CM_HighWide", focus, 0.5f, false, 10);
            director.hot = Cam("CM_Hot_Closeup", hot, 0.25f, true, 10);
            director.winner = Cam("CM_Winner", winner, 0.6f, false, 10, 0.22f);    // below the result card (CM3: +y = lower on screen)
            director.leaderClose = Cam("CM_Leader_Closeup", leaderT, 0.12f, false, 10, 0f);   // locked on (runners move 4 m/s)
            var closeLens = director.leaderClose.Lens;
            closeLens.FieldOfView = 42f;                                            // 3.2 m away: head to feet
            director.leaderClose.Lens = closeLens;
            director.leaderProxy = leaderT;
            director.focusProxy = focus;
            director.hotProxy = hot;
            director.winnerProxy = winner;
            director.tension = tension;
            // camera on the most relevant athletes (user, 2026-09-29): the current winner, and anyone close to falling
            // calm pacing (user: "too many camera cuts"): ≥ 4 s shots, winner close-ups, near falls ≥ 0.7 danger, 8 s apart
            director.hotCutDanger = 0.7f;
            director.minShotSeconds = 4f;
            director.hotCooldown = 8f;
            director.closeupSeconds = 8f;
            director.contextSeconds = 5f;
            director.leaderSwitchSeconds = 5f;
            EditorUtility.SetDirty(director);

            // impact FX pools
            var fxGo = Child(root, "ImpactFx");
            var fx = fxGo.AddComponent<ImpactFx>();
            var impulse = fxGo.AddComponent<CinemachineImpulseSource>();
            impulse.ImpulseDefinition.ImpulseShape = CinemachineImpulseDefinition.ImpulseShapes.Bump;
            impulse.ImpulseDefinition.ImpulseDuration = 0.25f;
            impulse.ImpulseDefinition.ImpactRadius = 200f;
            impulse.ImpulseDefinition.DissipationDistance = 400f;
            fx.impulse = impulse;
            fx.footDust = Pool<ParticleSystem>(fxGo, FxPrefabs + "/FootDust.prefab", 12);
            fx.slamDust = Pool<ParticleSystem>(fxGo, FxPrefabs + "/SlamDust.prefab", 4);
            fx.shockwaves = Pool<ParticleSystem>(fxGo, FxPrefabs + "/Shockwave.prefab", 4);
            fx.sparks = Pool<VisualEffect>(fxGo, FxPrefabs + "/SparksVfx.prefab", 3);
            fx.sparksFallback = Pool<ParticleSystem>(fxGo, FxPrefabs + "/SparksFallback.prefab", 3);
            fx.marks = Pool<ParticleSystem>(fxGo, FxPrefabs + "/ScuffMark.prefab", 12);
            fx.decals = Pool<DecalProjector>(fxGo, FxPrefabs + "/ScuffDecal.prefab", 16);
            foreach (var d in fx.decals) d.enabled = false;
            var pulse = Child(fxGo, "ImpactPulse").AddComponent<Volume>();
            pulse.isGlobal = true;
            pulse.priority = 10f;
            pulse.weight = 0f;
            pulse.sharedProfile = AssetDatabase.LoadAssetAtPath<VolumeProfile>(PulseProfile);
            fx.pulseVolume = pulse;

            // collision-shape view (GameSettings.ColliderView): draws from mjData, needs only meshes and a material
            var xray = Child(root, "ColliderXray").AddComponent<ColliderXray>();
            xray.sphere = Resources.GetBuiltinResource<Mesh>("Sphere.fbx");
            xray.cylinder = Resources.GetBuiltinResource<Mesh>("Cylinder.fbx");
            xray.cube = Resources.GetBuiltinResource<Mesh>("Cube.fbx");
            xray.material = AssetDatabase.LoadAssetAtPath<Material>(FxFolder + "/FxXray.mat");

            // follow spot on the athlete the shot is about (BroadcastDirector.UpdateLook aims and dims it; off at rest)
            var spot = Child(root, "FocusSpot").AddComponent<Light>();
            spot.type = LightType.Spot;
            spot.spotAngle = 26f;
            spot.innerSpotAngle = 14f;
            spot.range = 30f;
            spot.intensity = 0f;
            spot.color = new Color(1f, 0.96f, 0.88f);
            spot.shadows = LightShadows.None;
            spot.lightmapBakeType = LightmapBakeType.Realtime;
            spot.enabled = false;
            director.focusSpot = spot;
            EditorUtility.SetDirty(director);

            // audio
            var audioGo = Child(root, "ArenaAudio");
            var audio = audioGo.AddComponent<ArenaAudio>();
            audio.board = hud.board;
            audio.tension = tension;
            audio.bed = Source(audioGo, "CrowdBed", Clip("Crowd/crowd_bed_loop.ogg"), true, 0.35f);
            audio.tensionLayer = Source(audioGo, "CrowdTension", Clip("Crowd/crowd_tension_loop.ogg"), true, 0f);
            audio.stinger = Source(audioGo, "Stingers", null, false, 1f);
            audio.ui = Source(audioGo, "UiStings", null, false, 1f);
            AudioSource Source3D(string name, AudioClip clip, bool loop, float min, float max, float doppler = 0f)
            {
                var s = Source(audioGo, name, clip, loop, loop ? 0f : 1f);
                s.spatialBlend = 1f;
                s.rolloffMode = AudioRolloffMode.Linear;
                s.minDistance = min;
                s.maxDistance = max;
                s.dopplerLevel = doppler;
                return s;
            }
            // 8 voices: a hit on steel or glass plays two (the body + the surface)
            audio.voices = Enumerable.Range(0, 8).Select(i => Source3D($"Voice3D_{i}", null, false, 4f, 70f)).ToArray();
            audio.scrapes = Enumerable.Range(0, 3).Select(i => Source3D($"Scrape3D_{i}", Clip("Sfx/Scrape/scrape_loop.wav"), true, 3f, 50f)).ToArray();
            audio.whoosh = Source3D("CubeWhoosh3D", Clip("Sfx/Whoosh/whoosh_loop.wav"), true, 2f, 40f, 1f);
            audio.breath = Source3D("Breath3D", Clip("Sfx/Breath/breath_loop.wav"), true, 2f, 25f);
            audio.cubes = UnityEngine.Object.FindAnyObjectByType<MjCubePool>(FindObjectsInactive.Include);
            audio.metalHits = Clips("Sfx/Metal", "");
            audio.glassHits = Clips("Sfx/Glass", "");
            audio.cheers = Clips("Crowd", "crowd_cheer_big_");
            audio.reactions = Clips("Crowd", "crowd_react_");
            audio.bodySlams = Clips("Sfx/BodySlam", "");
            audio.cubeHits = Clips("Sfx/CubeHit", "");
            audio.thuds = Clips("Sfx/Thud", "");
            audio.footsteps = Clips("Sfx/Footsteps", "");
            audio.recordChime = Clip("Ui/confirmation_002.ogg");
            audio.goBong = Clip("Ui/bong_001.ogg");
            audio.countTick = Clip("Ui/click_002.ogg");

            // per athlete: telemetry, critic
            var conf = AssetDatabase.LoadAssetAtPath<TextAsset>(ConfidenceModel);
            foreach (var r in runners)
            {
                var tel = r.GetComponent<AthleteTelemetry>() ?? r.gameObject.AddComponent<AthleteTelemetry>();
                tel.confidenceModel = conf;
                var look = r.GetComponent<FatigueLook>();
                if (look == null) look = r.gameObject.AddComponent<FatigueLook>();
                look.skins = r.GetComponentsInChildren<SkinnedMeshRenderer>(true).Cast<Renderer>().ToArray();
                if (r.brain != null)
                {
                    var brainPath = AssetDatabase.GetAssetPath(r.brain);
                    var criticPath = Path.ChangeExtension(brainPath, null) + ".critic.onnx";
                    r.critic = AssetDatabase.LoadAssetAtPath<ModelAsset>(criticPath);
                    // fall recovery: this body's get-up brain, where one is deployed (role "getup" in the brain manifest)
                    var getup = $"{Path.GetDirectoryName(brainPath)}/getup_{Odds.BodyOf(r).ToLowerInvariant()}.onnx".Replace('\\', '/');
                    r.recoverBrain = AssetDatabase.LoadAssetAtPath<ModelAsset>(getup);
                    r.recoverSidecar = r.recoverBrain != null ? AssetDatabase.LoadAssetAtPath<TextAsset>(getup + ".json") : null;
                }
                EditorUtility.SetDirty(r);
            }

            hud.tension = tension;
            hud.arenaAudio = audio;
            EditorUtility.SetDirty(hud);

            // showcase: crowd director, stadium screens, medal ceremony, mixer + PA + crowd sectors, perf overlay
            StadiumShowcase.InstallBroadcast(hud, director, root, tension, audio, rig);
            StadiumLook.AthleteFill(scene);           // after InstallBroadcast: it creates the podium statues
            DressProps(scene);
        }

        public const string PropsAsset = "Assets/PoOlympic/Art/Props/EventProps.glb";

        /// <summary>Blender-built visuals (SourceArt/Props/build_props.py) for the small MuJoCo props a scene draws
        /// as bare boxes: pool cubes, slalom poles, trench posts, crab rails. The mesh hangs under its MjGeom as a
        /// child "Prop" (it moves with the geom) and the box renderer is switched off; the collision box is untouched.
        /// Only geoms that are drawn today and have the size the prop was modelled for (1 mm) are dressed.</summary>
        public static int DressProps(UnityEngine.SceneManagement.Scene scene)
        {
            var asset = AssetDatabase.LoadAssetAtPath<GameObject>(PropsAsset);
            if (asset == null) return 0;
            // geom name prefix → prop, half extents in Unity axes (MuJoCo x, z, y)
            var props = new (string prefix, string prop, Vector3 half)[]
            {
                ("cube", "PoolCube", new Vector3(0.1f, 0.1f, 0.1f)), ("pole", "SlalomPole", new Vector3(0.02f, 0.6f, 0.02f)),
                ("trench_post", "TrenchPost", new Vector3(0.025f, 0.36f, 0.025f)), ("rail", "CrabRail", new Vector3(0.02f, 0.02f, 10f)),
            };
            int dressed = 0;
            foreach (var geom in scene.GetRootGameObjects().SelectMany(g => g.GetComponentsInChildren<Mujoco.MjGeom>(true)))
            {
                var p = props.FirstOrDefault(x => geom.name.StartsWith(x.prefix));
                if (p.prop == null || geom.ShapeType != Mujoco.MjShapeComponent.ShapeTypes.Box || (geom.Box.Extents - p.half).magnitude > 0.001f) continue;
                var source = asset.GetComponentsInChildren<MeshFilter>(true).FirstOrDefault(f => f.name == p.prop);
                var old = geom.transform.Find("Prop");
                if (source == null || !geom.TryGetComponent<MeshRenderer>(out var box) || (!box.enabled && old == null)) continue;
                var go = old != null ? old.gameObject : Child(geom.gameObject, "Prop");
                // (no ??: GetComponent returns a fake null in the editor)
                if (!go.TryGetComponent<MeshFilter>(out var filter)) filter = go.AddComponent<MeshFilter>();
                if (!go.TryGetComponent<MeshRenderer>(out var rend)) rend = go.AddComponent<MeshRenderer>();
                filter.sharedMesh = source.sharedMesh;
                rend.sharedMaterials = source.GetComponent<MeshRenderer>().sharedMaterials;
                box.enabled = false;
                dressed++;
            }
            return dressed;
        }

        static GameObject Child(GameObject parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            return go;
        }

        static T[] Pool<T>(GameObject parent, string prefabPath, int n) where T : Component
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) ?? throw new FileNotFoundException(prefabPath);
            var holder = Child(parent, Path.GetFileNameWithoutExtension(prefabPath) + "Pool");
            return Enumerable.Range(0, n).Select(i =>
            {
                var inst = (GameObject)PrefabUtility.InstantiatePrefab(prefab, holder.transform);
                inst.name = $"{prefab.name}_{i}";
                inst.transform.localPosition = Vector3.down * 50f;          // parked out of sight until used
                return inst.GetComponent<T>();
            }).ToArray();
        }

        static AudioSource Source(GameObject parent, string name, AudioClip clip, bool loop, float volume)
        {
            var s = Child(parent, name).AddComponent<AudioSource>();
            s.clip = clip;
            s.loop = loop;
            s.playOnAwake = false;
            s.volume = volume;
            s.spatialBlend = 0f;
            return s;
        }

        static AudioClip Clip(string rel) => AssetDatabase.LoadAssetAtPath<AudioClip>($"{AudioFolder}/{rel}");

        static AudioClip[] Clips(string folder, string prefix) =>
            AssetDatabase.FindAssets("t:AudioClip", new[] { $"{AudioFolder}/{folder}" })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Where(p => Path.GetDirectoryName(p).Replace('\\', '/') == $"{AudioFolder}/{folder}" && Path.GetFileName(p).StartsWith(prefix))
                .OrderBy(p => p).Select(AssetDatabase.LoadAssetAtPath<AudioClip>).ToArray();

        // ------------------------------------------------------------------------------------------------ assets
        /// <summary>Textures, materials, effect prefabs and audio import settings (created once, kept if present).</summary>
        public static void EnsureAssets()
        {
            Directory.CreateDirectory(FxFolder);
            Directory.CreateDirectory(FxPrefabs);
            var dot = Texture(FxFolder + "/fx_soft_dot.png", 64, (x, y) => Mathf.Pow(Mathf.Clamp01(1f - Mathf.Sqrt(x * x + y * y)), 1.6f));
            var ring = Texture(FxFolder + "/fx_ring.png", 128, (x, y) =>
            {
                float r = Mathf.Sqrt(x * x + y * y);
                return Mathf.Clamp01(1f - Mathf.Abs(r - 0.82f) / 0.12f) * (r < 1f ? 1f : 0f);
            });
            var dust = ParticleMaterial("FxDust", dot, false);
            var shock = ParticleMaterial("FxShock", ring, true);
            var spark = ParticleMaterial("FxSpark", dot, true);
            // scuff: a soft blot with streaks along x, dark on the track
            var scuffTex = Texture(FxFolder + "/fx_scuff.png", 128, (x, y) =>
                Mathf.Pow(Mathf.Clamp01(1f - Mathf.Sqrt(x * x * 0.6f + y * y * 2.2f)), 0.8f) * (0.55f + 0.45f * Mathf.Sin(y * 23f) * Mathf.Sin(x * 5f + y * 9f)));
            var scuff = ParticleMaterial("FxScuff", scuffTex, false);
            ScrapeClip();
            SurfaceClips();
            AudioImport();
            EnsureDecalFeature();
            EnsurePulseProfile();

            // projected scuff: the same blot, dark in the texture itself (the decal shader has no tint)
            var decalTex = Texture(FxFolder + "/fx_scuff_decal.png", 128, (x, y) =>
                0.55f * Mathf.Pow(Mathf.Clamp01(1f - Mathf.Sqrt(x * x * 0.6f + y * y * 2.2f)), 0.8f) * (0.55f + 0.45f * Mathf.Sin(y * 23f) * Mathf.Sin(x * 5f + y * 9f)),
                new Color(0.07f, 0.06f, 0.05f));
            var decalMat = ShaderMaterial("FxScuffDecal", "Shader Graphs/Decal");
            decalMat.SetTexture("Base_Map", decalTex);
            decalMat.SetFloat("Normal_Blend", 0f);
            EditorUtility.SetDirty(decalMat);
            MakePrefab("ScuffDecal", go =>
            {
                var d = go.AddComponent<DecalProjector>();
                d.material = decalMat;
                d.scaleMode = DecalScaleMode.InheritFromHierarchy;       // ImpactFx sizes a mark by the transform
                d.size = new Vector3(0.5f, 0.5f, 0.1f);
                d.pivot = Vector3.zero;                                   // ± 5 cm round the contact point
                d.drawDistance = 60f;
                d.startAngleFade = 60f;
                d.endAngleFade = 85f;
            });
            ShaderMaterial("FxXray", "PoOlympic/Xray");

            MakePrefab("FootDust", go =>
            {
                var ps = Particles(go, dust, 0.3f, (0.45f, 0.75f), (0.4f, 1.1f), (0.12f, 0.26f), new Color(0.66f, 0.58f, 0.5f, 0.55f), 40);
                var main = ps.main; main.gravityModifier = -0.04f;
                var em = ps.emission; em.SetBursts(new[] { new ParticleSystem.Burst(0f, 9) });
                var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Hemisphere; sh.radius = 0.06f; sh.rotation = new Vector3(-90f, 0f, 0f);
                Grow(ps, 2.2f);
                FadeOut(ps);
            });
            MakePrefab("SlamDust", go =>
            {
                var ps = Particles(go, dust, 0.3f, (0.6f, 1.1f), (1.4f, 3.4f), (0.22f, 0.5f), new Color(0.6f, 0.54f, 0.47f, 0.6f), 60);
                var em = ps.emission; em.SetBursts(new[] { new ParticleSystem.Burst(0f, 28) });
                var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Circle; sh.radius = 0.18f; sh.rotation = new Vector3(90f, 0f, 0f);
                var lim = ps.limitVelocityOverLifetime; lim.enabled = true; lim.dampen = 0.12f; lim.limit = 0.3f;
                Grow(ps, 2.6f);
                FadeOut(ps);
            });
            MakePrefab("Shockwave", go =>
            {
                var ps = Particles(go, shock, 0.1f, (0.42f, 0.42f), (0f, 0f), (0.5f, 0.5f), new Color(1f, 0.95f, 0.85f, 0.75f), 2);
                var em = ps.emission; em.SetBursts(new[] { new ParticleSystem.Burst(0f, 1) });
                var sh = ps.shape; sh.enabled = false;
                go.GetComponent<ParticleSystemRenderer>().renderMode = ParticleSystemRenderMode.HorizontalBillboard;
                Grow(ps, 6f);
                FadeOut(ps);
            });
            MakePrefab("SparksFallback", go =>
            {
                var ps = Particles(go, spark, 0.1f, (0.22f, 0.5f), (3f, 7f), (0.025f, 0.05f), new Color(1f, 0.72f, 0.28f, 1f), 60);
                var main = ps.main; main.gravityModifier = 1.3f;
                var em = ps.emission; em.SetBursts(new[] { new ParticleSystem.Burst(0f, 30) });
                var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Sphere; sh.radius = 0.05f;
                var r = go.GetComponent<ParticleSystemRenderer>();
                r.renderMode = ParticleSystemRenderMode.Stretch;
                r.velocityScale = 0.06f;
                r.lengthScale = 1f;
                FadeOut(ps);
            });
            // ground mark: one flat particle that stays ~18 s and fades (world space, so a pooled emitter can move on
            // and leave its earlier marks where they are)
            MakePrefab("ScuffMark", go =>
            {
                var ps = Particles(go, scuff, 0.1f, (16f, 20f), (0f, 0f), (0.35f, 0.6f), new Color(0.07f, 0.06f, 0.05f, 0.5f), 6);
                var em = ps.emission; em.SetBursts(new[] { new ParticleSystem.Burst(0f, 1) });
                var sh = ps.shape; sh.enabled = false;
                go.GetComponent<ParticleSystemRenderer>().renderMode = ParticleSystemRenderMode.HorizontalBillboard;
                var c = ps.colorOverLifetime;
                c.enabled = true;
                var g = new Gradient();
                g.SetKeys(new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                          new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(1f, 0.7f), new GradientAlphaKey(0f, 1f) });
                c.color = g;
            });
            var sparks = AssetDatabase.LoadAssetAtPath<VisualEffectAsset>(SparksVfx);
            MakePrefab("SparksVfx", go =>
            {
                var v = go.AddComponent<VisualEffect>();
                v.visualEffectAsset = sparks;
                v.initialEventName = "";                 // idle until ImpactFx plays it
            });
            AssetDatabase.SaveAssets();
        }

        static Material ShaderMaterial(string name, string shader)
        {
            var path = $"{FxFolder}/{name}.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat != null) return mat;
            mat = new Material(Shader.Find(shader) ?? throw new InvalidOperationException($"shader '{shader}' not found")) { name = name };
            AssetDatabase.CreateAsset(mat, path);
            return mat;
        }

        /// <summary>The URP Decal renderer feature on both renderers (PC, Mobile): DecalProjectors draw nothing without it.</summary>
        static void EnsureDecalFeature()
        {
            foreach (var path in new[] { "Assets/Settings/PC_Renderer.asset", "Assets/Settings/Mobile_Renderer.asset" })
            {
                var data = AssetDatabase.LoadAssetAtPath<ScriptableRendererData>(path);
                if (data == null || data.rendererFeatures.Any(f => f is DecalRendererFeature)) continue;
                var feature = ScriptableObject.CreateInstance<DecalRendererFeature>();
                feature.name = "Decal";
                AssetDatabase.AddObjectToAsset(feature, data);
                AssetDatabase.TryGetGUIDAndLocalFileIdentifier(feature, out _, out long id);
                // the renderer keeps the list and a parallel map of local file ids (what its own inspector writes)
                var so = new SerializedObject(data);
                var list = so.FindProperty("m_RendererFeatures");
                var map = so.FindProperty("m_RendererFeatureMap");
                list.arraySize++;
                list.GetArrayElementAtIndex(list.arraySize - 1).objectReferenceValue = feature;
                map.arraySize++;
                map.GetArrayElementAtIndex(map.arraySize - 1).longValue = id;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(data);
            }
        }

        public const string PulseProfile = "Assets/PoOlympic/Settings/ImpactPulse_Volume.asset";

        /// <summary>What a big slam flashes (ImpactFx.pulseVolume, weight 0 → 1 → 0): colour fringe and darker corners.
        /// No lens distortion: the picture must not move (user, 2026-09-30: "fast movements hurt my eyes").</summary>
        static void EnsurePulseProfile()
        {
            if (AssetDatabase.LoadAssetAtPath<VolumeProfile>(PulseProfile) != null) return;
            var p = ScriptableObject.CreateInstance<VolumeProfile>();
            AssetDatabase.CreateAsset(p, PulseProfile);
            var ca = p.Add<ChromaticAberration>(true);
            ca.intensity.Override(0.7f);
            var vig = p.Add<Vignette>(true);
            vig.intensity.Override(0.45f);
            vig.smoothness.Override(0.4f);
            foreach (var c in p.components) AssetDatabase.AddObjectToAsset(c, p);
            EditorUtility.SetDirty(p);
        }

        static Texture2D Texture(string path, int size, Func<float, float, float> alpha, Color? rgb = null)
        {
            var existing = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (existing != null) return existing;
            var t = new Texture2D(size, size, TextureFormat.RGBA32, false);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float u = (x + 0.5f) / size * 2f - 1f, v = (y + 0.5f) / size * 2f - 1f;
                    var c = rgb ?? Color.white;
                    c.a = alpha(u, v);
                    t.SetPixel(x, y, c);
                }
            File.WriteAllBytes(path, t.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(t);
            AssetDatabase.ImportAsset(path);
            var imp = (TextureImporter)AssetImporter.GetAtPath(path);
            imp.alphaIsTransparency = true;
            imp.mipmapEnabled = false;
            imp.wrapMode = TextureWrapMode.Clamp;
            imp.SaveAndReimport();
            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        /// <summary>URP Particles/Unlit, transparent (alpha or additive), vertex colour × texture.</summary>
        static Material ParticleMaterial(string name, Texture2D tex, bool additive)
        {
            var path = $"{FxFolder}/{name}.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (mat == null)
            {
                mat = new Material(Shader.Find("Universal Render Pipeline/Particles/Unlit")) { name = name };
                AssetDatabase.CreateAsset(mat, path);
            }
            if (tex != null) mat.SetTexture("_BaseMap", tex);
            mat.SetColor("_BaseColor", Color.white);
            mat.SetFloat("_Surface", 1f);
            mat.SetFloat("_Blend", additive ? 2f : 0f);
            mat.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetFloat("_DstBlend", (float)(additive ? UnityEngine.Rendering.BlendMode.One : UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha));
            mat.SetFloat("_SrcBlendAlpha", (float)UnityEngine.Rendering.BlendMode.One);
            mat.SetFloat("_DstBlendAlpha", (float)(additive ? UnityEngine.Rendering.BlendMode.One : UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha));
            mat.SetFloat("_ZWrite", 0f);
            mat.SetFloat("_Cull", 0f);
            mat.SetOverrideTag("RenderType", "Transparent");
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            EditorUtility.SetDirty(mat);
            return mat;
        }

        static void MakePrefab(string name, Action<GameObject> build)
        {
            var path = $"{FxPrefabs}/{name}.prefab";
            var go = new GameObject(name);
            try
            {
                build(go);
                PrefabUtility.SaveAsPrefabAsset(go, path);
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        static ParticleSystem Particles(GameObject go, Material mat, float duration, (float, float) life, (float, float) speed,
                                        (float, float) size, Color color, int max)
        {
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            var main = ps.main;
            main.duration = duration;
            main.loop = false;
            main.playOnAwake = false;
            main.startLifetime = new ParticleSystem.MinMaxCurve(life.Item1, life.Item2);
            main.startSpeed = new ParticleSystem.MinMaxCurve(speed.Item1, speed.Item2);
            main.startSize = new ParticleSystem.MinMaxCurve(size.Item1, size.Item2);
            main.startRotation = new ParticleSystem.MinMaxCurve(0f, Mathf.PI * 2f);
            main.startColor = color;
            main.maxParticles = max;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.scalingMode = ParticleSystemScalingMode.Hierarchy;
            var em = ps.emission;
            em.rateOverTime = 0f;
            var r = go.GetComponent<ParticleSystemRenderer>();
            r.sharedMaterial = mat;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
            r.sortMode = ParticleSystemSortMode.Distance;
            return ps;
        }

        static void Grow(ParticleSystem ps, float to)
        {
            var s = ps.sizeOverLifetime;
            s.enabled = true;
            s.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.EaseInOut(0f, 1f / to, 1f, 1f));
            s.sizeMultiplier = to;
        }

        static void FadeOut(ParticleSystem ps)
        {
            var c = ps.colorOverLifetime;
            c.enabled = true;
            var g = new Gradient();
            g.SetKeys(new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                      new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(0.8f, 0.3f), new GradientAlphaKey(0f, 1f) });
            c.color = g;
        }

        /// <summary>The surface-drag loop (ArenaAudio.Scrape): 1 s of band-limited noise with a little grain, generated
        /// once as a 22 kHz mono WAV whose end is cross-faded into its start (no click at the loop point).</summary>
        static void ScrapeClip()
        {
            var path = AudioFolder + "/Sfx/Scrape/scrape_loop.wav";
            if (File.Exists(path)) return;
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            const int rate = 22050, n = rate, fade = rate / 10;
            var rng = new System.Random(11);
            var raw = new float[n + fade];
            float lp = 0f, slow = 0f;
            for (int i = 0; i < raw.Length; i++)
            {
                float white = (float)rng.NextDouble() * 2f - 1f;
                lp += 0.32f * (white - lp);                 // roll off the hiss
                slow += 0.015f * (lp - slow);               // and take out the rumble: what is left is the "shhh"
                raw[i] = (lp - slow) * (0.75f + 0.25f * Mathf.Sin(2f * Mathf.PI * 37f * i / rate));
            }
            using var w = new BinaryWriter(File.Create(path));
            w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + n * 2); w.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
            w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16);
            w.Write(System.Text.Encoding.ASCII.GetBytes("data")); w.Write(n * 2);
            for (int i = 0; i < n; i++)
            {
                float u = i < fade ? (float)i / fade : 1f;
                float v = raw[i] * u + (i < fade ? raw[n + i] * (1f - u) : 0f);
                w.Write((short)Mathf.Clamp(v * 2.2f * 32767f, -32767f, 32767f));
            }
            w.Close();
            AssetDatabase.ImportAsset(path);
        }

        /// <summary>The generated surface and effort sounds (22 kHz mono, written once; docs/LICENSING.md "generated"):
        ///   Sfx/Metal   steel ring: four bar-mode partials decaying over ~0.4 s, three pitches
        ///   Sfx/Glass   glass knock: the same, higher and much shorter
        ///   Sfx/Whoosh  air past a thrown cube: smooth band noise loop
        ///   Sfx/Breath  one in-and-out breath (1.2 s) as a loop: noise shaped by the two phases</summary>
        static void SurfaceClips()
        {
            const int rate = 22050;
            float[] Ring(float f0, float seconds, float decay)
            {
                var s = new float[(int)(rate * seconds)];
                float[] ratio = { 1f, 2.76f, 5.4f, 8.93f }, gain = { 1f, 0.6f, 0.35f, 0.2f };       // free bar modes
                for (int i = 0; i < s.Length; i++)
                {
                    float t = (float)i / rate, v = 0f;
                    for (int k = 0; k < ratio.Length; k++)
                        v += gain[k] * Mathf.Sin(2f * Mathf.PI * f0 * ratio[k] * t) * Mathf.Exp(-t * decay * (1f + k));
                    s[i] = v * 0.4f * Mathf.Min(1f, i / 30f);                                        // 1.4 ms attack: no click
                }
                return s;
            }
            float[] Noise(float seconds, int seed, float hiss, float rumble, Func<float, float> envelope)
            {
                int n = (int)(rate * seconds), fade = rate / 10;
                var rng = new System.Random(seed);
                var raw = new float[n + fade];
                float lp = 0f, slow = 0f;
                for (int i = 0; i < raw.Length; i++)
                {
                    lp += hiss * ((float)rng.NextDouble() * 2f - 1f - lp);
                    slow += rumble * (lp - slow);
                    raw[i] = (lp - slow) * envelope((float)(i % n) / n);
                }
                var s = new float[n];                                                                // end cross-faded into the start
                for (int i = 0; i < n; i++)
                {
                    float u = i < fade ? (float)i / fade : 1f;
                    s[i] = Mathf.Clamp(2.2f * (raw[i] * u + (i < fade ? raw[n + i] * (1f - u) : 0f)), -1f, 1f);
                }
                return s;
            }
            for (int i = 0; i < 3; i++)
            {
                Wav($"{AudioFolder}/Sfx/Metal/metal_ring_{i}.wav", rate, () => Ring(430f + 90f * i, 0.45f, 9f));
                Wav($"{AudioFolder}/Sfx/Glass/glass_knock_{i}.wav", rate, () => Ring(1250f + 240f * i, 0.16f, 34f));
            }
            Wav($"{AudioFolder}/Sfx/Whoosh/whoosh_loop.wav", rate, () => Noise(1f, 23, 0.12f, 0.02f, _ => 1f));
            // in (0 – 40 % of the loop, thinner) … pause … out (50 – 90 %, fuller)
            Wav($"{AudioFolder}/Sfx/Breath/breath_loop.wav", rate, () => Noise(1.2f, 31, 0.2f, 0.03f, u =>
                u < 0.4f ? 0.6f * Mathf.Sin(Mathf.PI * u / 0.4f) : u > 0.5f && u < 0.9f ? Mathf.Sin(Mathf.PI * (u - 0.5f) / 0.4f) : 0f));
        }

        /// <summary>16-bit mono WAV, written only if the file is not there yet.</summary>
        static void Wav(string path, int rate, Func<float[]> samples)
        {
            if (File.Exists(path)) return;
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            var s = samples();
            using (var w = new BinaryWriter(File.Create(path)))
            {
                w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + s.Length * 2); w.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
                w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16);
                w.Write(System.Text.Encoding.ASCII.GetBytes("data")); w.Write(s.Length * 2);
                foreach (var v in s) w.Write((short)Mathf.Clamp(v * 32767f, -32767f, 32767f));
            }
            AssetDatabase.ImportAsset(path);
        }

        /// <summary>Mobile-friendly import: crowd loops stereo, compressed in memory; short SFX mono (3D), decompressed on load.</summary>
        static void AudioImport()
        {
            foreach (var guid in AssetDatabase.FindAssets("t:AudioClip", new[] { AudioFolder }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var imp = (AudioImporter)AssetImporter.GetAtPath(path);
                bool crowd = path.Contains("/Crowd/");
                var s = imp.defaultSampleSettings;
                s.loadType = crowd ? AudioClipLoadType.CompressedInMemory : AudioClipLoadType.DecompressOnLoad;
                s.compressionFormat = AudioCompressionFormat.Vorbis;
                s.quality = crowd ? 0.6f : 0.7f;
                bool changed = imp.forceToMono != !crowd || imp.defaultSampleSettings.loadType != s.loadType;
                imp.forceToMono = !crowd;
                imp.loadInBackground = crowd;
                imp.defaultSampleSettings = s;
                if (changed) imp.SaveAndReimport();
            }
        }
    }
}
