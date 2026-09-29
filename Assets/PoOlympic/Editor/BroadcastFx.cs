using System;
using System.IO;
using System.Linq;
using Unity.Cinemachine;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
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
    ///   per athlete                   AthleteTelemetry, the brain's critic (&lt;brain&gt;.critic.onnx), BalanceOverlay
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
            var noise = AssetDatabase.LoadAssetAtPath<NoiseSettings>(NoiseProfile);
            var lens = LensSettings.FromCamera(cam);
            CinemachineCamera Cam(string name, Transform target, float damping, bool handheld, int priority)
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
                // 9:16 frame with the standings on top: keep athletes a little below the centre
                var comp = aim.Composition;
                comp.ScreenPosition = new Vector2(0f, -0.06f);
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
            director.winner = Cam("CM_Winner", winner, 0.6f, false, 10);
            director.focusProxy = focus;
            director.hotProxy = hot;
            director.winnerProxy = winner;
            director.tension = tension;
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

            // audio
            var audioGo = Child(root, "ArenaAudio");
            var audio = audioGo.AddComponent<ArenaAudio>();
            audio.board = hud.board;
            audio.tension = tension;
            audio.bed = Source(audioGo, "CrowdBed", Clip("Crowd/crowd_bed_loop.ogg"), true, 0.35f);
            audio.tensionLayer = Source(audioGo, "CrowdTension", Clip("Crowd/crowd_tension_loop.ogg"), true, 0f);
            audio.stinger = Source(audioGo, "Stingers", null, false, 1f);
            audio.ui = Source(audioGo, "UiStings", null, false, 1f);
            audio.voices = Enumerable.Range(0, 6).Select(i =>
            {
                var s = Source(audioGo, $"Voice3D_{i}", null, false, 1f);
                s.spatialBlend = 1f;
                s.rolloffMode = AudioRolloffMode.Linear;
                s.minDistance = 4f;
                s.maxDistance = 70f;
                s.dopplerLevel = 0f;
                return s;
            }).ToArray();
            audio.cheers = Clips("Crowd", "crowd_cheer_big_");
            audio.reactions = Clips("Crowd", "crowd_react_");
            audio.bodySlams = Clips("Sfx/BodySlam", "");
            audio.cubeHits = Clips("Sfx/CubeHit", "");
            audio.thuds = Clips("Sfx/Thud", "");
            audio.footsteps = Clips("Sfx/Footsteps", "");
            audio.recordChime = Clip("Ui/confirmation_002.ogg");
            audio.goBong = Clip("Ui/bong_001.ogg");
            audio.countTick = Clip("Ui/click_002.ogg");

            // per athlete: telemetry, critic, balance overlay
            var conf = AssetDatabase.LoadAssetAtPath<TextAsset>(ConfidenceModel);
            var overlayMat = AssetDatabase.LoadAssetAtPath<Material>(FxFolder + "/FxOverlay.mat");
            foreach (var r in runners)
            {
                var tel = r.GetComponent<AthleteTelemetry>() ?? r.gameObject.AddComponent<AthleteTelemetry>();
                tel.confidenceModel = conf;
                if (r.brain != null)
                {
                    var brainPath = AssetDatabase.GetAssetPath(r.brain);
                    var criticPath = Path.ChangeExtension(brainPath, null) + ".critic.onnx";
                    r.critic = AssetDatabase.LoadAssetAtPath<ModelAsset>(criticPath);
                }
                foreach (var old in r.GetComponentsInChildren<BalanceOverlay>(true)) UnityEngine.Object.DestroyImmediate(old.gameObject);
                var ov = new GameObject("BalanceOverlay");
                ov.transform.SetParent(r.transform, false);
                var o = ov.AddComponent<BalanceOverlay>();
                o.telemetry = tel;
                o.showAlways = director.kind == BroadcastDirector.Kind.Arena;
                o.hull = Line(ov, "SupportPolygon", overlayMat, 0.035f, true);
                o.ring = Line(ov, "ComRing", overlayMat, 0.03f, true);
                o.plumb = Line(ov, "PlumbLine", overlayMat, 0.02f, false);
                EditorUtility.SetDirty(r);
            }

            hud.tension = tension;
            hud.arenaAudio = audio;
            EditorUtility.SetDirty(hud);
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

        static LineRenderer Line(GameObject parent, string name, Material mat, float width, bool flat)
        {
            var go = Child(parent, name);
            if (flat) go.transform.rotation = Quaternion.Euler(90f, 0f, 0f);      // TransformZ alignment → lies on the ground
            var l = go.AddComponent<LineRenderer>();
            l.sharedMaterial = mat;
            l.useWorldSpace = true;
            l.widthMultiplier = width;
            l.alignment = flat ? LineAlignment.TransformZ : LineAlignment.View;
            l.numCornerVertices = 2;
            l.numCapVertices = 2;
            l.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            l.receiveShadows = false;
            l.positionCount = 0;
            l.enabled = false;
            return l;
        }

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
            ParticleMaterial("FxOverlay", null, false);
            AudioImport();

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
            var sparks = AssetDatabase.LoadAssetAtPath<VisualEffectAsset>(SparksVfx);
            MakePrefab("SparksVfx", go =>
            {
                var v = go.AddComponent<VisualEffect>();
                v.visualEffectAsset = sparks;
                v.initialEventName = "";                 // idle until ImpactFx plays it
            });
            AssetDatabase.SaveAssets();
        }

        static Texture2D Texture(string path, int size, Func<float, float, float> alpha)
        {
            var existing = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (existing != null) return existing;
            var t = new Texture2D(size, size, TextureFormat.RGBA32, false);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float u = (x + 0.5f) / size * 2f - 1f, v = (y + 0.5f) / size * 2f - 1f;
                    t.SetPixel(x, y, new Color(1f, 1f, 1f, alpha(u, v)));
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
