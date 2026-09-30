using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.Cinemachine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Audio;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.UIElements;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Unity half of the stadium showcase pass (Blender half: SourceArt/Stadium/build_showcase.py) — the GFX/sound
    /// top 10 of 2026-09-29. Everything is authored into the scenes (editable in the editor), render / audio only:
    ///   stadium (Dress)   crowd strips → PoOlympic/Crowd (living crowd), beam cones → PoOlympic/LightBeam, Screen_Live
    ///                     → the live board render texture (cube: centre crop), no shadows from the rigging / cube /
    ///                     beams, GI flags for the bake, light probe grid over the event area
    ///   bake              LightingSettings (Settings/ArenaLighting.lighting): baked indirect + the 36 arena spots
    ///                     baked (athletes get them from the probes); the key light stays realtime (shadows)
    ///   broadcast         CrowdDirector, StadiumScreens (ScreenFeed + feed camera), PodiumCeremony (+ CM_Podium,
    ///                     confetti), audio mixer (Audio/PoOlympicMix.mixer, groups + snapshots), crowd sectors, PA
    ///                     announcer with echo, event SFX, arena reverb zone, hit-stop low-pass, PerfOverlay
    /// Menu: PoOlympic › Stadium › Upgrade showcase (all scenes) — re-dresses every built stadium scene, re-installs the
    /// broadcast layer; then bake each scene (Bake arena lighting (open scene) or unity command bake_lighting).
    /// </summary>
    public static class StadiumShowcase
    {
        const string Mats = "Assets/PoOlympic/Materials/Showcase";
        const string BoardRT = Mats + "/ScreenBoard.renderTexture", FeedRT = Mats + "/ScreenFeed.renderTexture";
        const string ScreenPanel = "Assets/PoOlympic/UI/ScreenPanelSettings.asset";
        const string BoardStyle = "Assets/PoOlympic/UI/ScreenBoard.uss", PerfStyle = "Assets/PoOlympic/UI/PerfOverlay.uss";
        const string Theme = "Assets/PoOlympic/UI/PoOlympicTheme.tss";
        const string MixerPath = "Assets/PoOlympic/Audio/PoOlympicMix.mixer";
        const string LightingPath = "Assets/PoOlympic/Settings/ArenaLighting.lighting";
        const string ConfettiPrefab = BroadcastFx.FxPrefabs + "/Confetti.prefab";
        public const float CubeCropU = 840f / 1280f;          // the video cube shows the centre 840 px of the board

        // ------------------------------------------------------------------------------------------------ menu
        [MenuItem("PoOlympic/Stadium/Upgrade showcase (all scenes)")]
        public static string UpgradeAll()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            var sb = new System.Text.StringBuilder();
            foreach (var guid in AssetDatabase.FindAssets("t:Scene", new[] { "Assets/PoOlympic/Scenes" }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                var r = UpgradeOpenScene();
                if (r == null) continue;
                EditorSceneManager.SaveScene(scene);
                sb.AppendLine($"{Path.GetFileNameWithoutExtension(path)}: {r}");
            }
            AssetDatabase.SaveAssets();
            return sb.ToString();
        }

        [MenuItem("PoOlympic/Stadium/Upgrade showcase (open scene)")]
        public static string UpgradeOpenScene()
        {
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            var stadium = scene.GetRootGameObjects().FirstOrDefault(g => g.name == "Stadium");
            if (stadium == null) return null;
            bool broadcast = UnityEngine.Object.FindAnyObjectByType<BroadcastHud>(FindObjectsInactive.Include) != null;
            StadiumLook.ArenaLighting(stadium);
            ApplyLightingSettings(scene);             // ArenaLighting recreates the spots as realtime: keep them baked
            var dressed = Dress(stadium, broadcast);
            string fx = broadcast ? BroadcastFx.InstallInOpenScene() : "no broadcast";
            EditorSceneManager.MarkSceneDirty(scene);
            return $"{dressed}; {fx}";
        }

        [MenuItem("PoOlympic/Stadium/Bake arena lighting (open scene)")]
        public static string BakeOpenScene()
        {
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            ApplyLightingSettings(scene);
            return Lightmapping.BakeAsync() ? "baking " + scene.name : "bake did not start";
        }

        // ------------------------------------------------------------------------------------------------ stadium
        /// <summary>Materials, shadows, GI flags and light probes of a placed stadium. `liveScreens`: the scene has a
        /// broadcast (ScreenFeed renders the board); elsewhere the screens keep their glTF material.</summary>
        public static string Dress(GameObject stadium, bool liveScreens)
        {
            EnsureAssets();
            var crowdMats = new Dictionary<string, Material>();
            foreach (var n in new[] { "Crowd_Blue", "Crowd_White" })          // refresh textures of already-dressed scenes
                if (AssetDatabase.LoadAssetAtPath<Material>($"{Mats}/{n}_Live.mat") != null) CrowdMaterial(n, null);
            var beam = AssetDatabase.LoadAssetAtPath<Material>(Mats + "/LightBeam.mat");
            var wide = AssetDatabase.LoadAssetAtPath<Material>(Mats + "/ScreenLive_Wide.mat");
            var cube = AssetDatabase.LoadAssetAtPath<Material>(Mats + "/ScreenLive_Cube.mat");
            int crowd = 0, beams = 0, screens = 0;
            foreach (var r in stadium.GetComponentsInChildren<Renderer>(true))
            {
                var mats = r.sharedMaterials;
                bool changed = false;
                for (int i = 0; i < mats.Length; i++)
                {
                    var m = mats[i];
                    if (m == null) continue;
                    string n = m.name.Replace(" (Instance)", "");
                    if (n == "Crowd_Blue" || n == "Crowd_White")
                    {
                        if (!crowdMats.TryGetValue(n, out var cm)) crowdMats[n] = cm = CrowdMaterial(n, m);
                        mats[i] = cm; changed = true; crowd++;
                    }
                    else if (n == "Light_Beam") { mats[i] = beam; changed = true; beams++; }
                    else if (n == "Screen_Live" && liveScreens) { mats[i] = r.name.StartsWith("VideoCube") || HasParent(r.transform, "VideoCube") ? cube : wide; changed = true; screens++; }
                }
                if (changed) { r.sharedMaterials = mats; EditorUtility.SetDirty(r); }
                string on = r.name;
                if (on.StartsWith("Light_Beams") || HasParent(r.transform, "Light_Beams"))
                {
                    r.shadowCastingMode = ShadowCastingMode.Off;
                    r.receiveShadows = false;
                }
                if (on.StartsWith("VideoCube") || on.StartsWith("Roof_Rigging") || HasParent(r.transform, "VideoCube") || HasParent(r.transform, "Roof_Rigging"))
                    r.shadowCastingMode = ShadowCastingMode.Off;
            }
            // podium flags stay down (inactive) until PodiumCeremony raises them
            foreach (var t in stadium.GetComponentsInChildren<Transform>(true))
                if (t.name.StartsWith("Podium_Flag_")) t.gameObject.SetActive(false);
            GiFlags(stadium);
            int probes = LightProbes(stadium);
            return $"crowd {crowd}, beams {beams}, screens {screens}, probes {probes}";
        }

        static bool HasParent(Transform t, string prefix)
        {
            for (var p = t.parent; p != null; p = p.parent) if (p.name.StartsWith(prefix)) return true;
            return false;
        }

        static Material CrowdMaterial(string name, Material gltf)
        {
            var path = $"{Mats}/{name}_Live.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null)
            {
                m = new Material(Shader.Find("PoOlympic/Crowd") ?? throw new InvalidOperationException("PoOlympic/Crowd shader missing"));
                AssetDatabase.CreateAsset(m, path);
            }
            // the two halves as separate project textures (build_showcase.py writes Art/Stadium/Crowd/*.png); the glTF
            // atlas is the fallback
            var seated = CrowdTexture($"{CrowdDir}/{name}_seated.png");
            var cheer = CrowdTexture($"{CrowdDir}/{name}_cheer.png");
            var tex = gltf == null ? m.GetTexture("_BaseMap") : gltf.HasProperty("baseColorTexture") ? gltf.GetTexture("baseColorTexture") : gltf.mainTexture;
            m.SetTexture("_BaseMap", seated != null ? seated : tex);
            m.SetTexture("_CheerMap", cheer != null ? cheer : seated != null ? seated : tex);
            EditorUtility.SetDirty(m);
            return m;
        }

        const string CrowdDir = "Assets/PoOlympic/Art/Stadium/Crowd";

        /// <summary>U repeats (64 seats per UV unit along the row), V clamps (tread → top of the card), trilinear.</summary>
        static Texture2D CrowdTexture(string path)
        {
            var imp = AssetImporter.GetAtPath(path) as TextureImporter;
            if (imp == null) return null;
            if (imp.wrapModeV != TextureWrapMode.Clamp || imp.anisoLevel != 4 || !imp.mipMapsPreserveCoverage)
            {
                imp.alphaIsTransparency = true;                  // dilated colour under the cut-outs (no dark fringes)
                imp.mipMapsPreserveCoverage = true;              // fans do not melt away in the far stands
                imp.alphaTestReferenceValue = 0.5f;
                imp.wrapModeU = TextureWrapMode.Repeat;
                imp.wrapModeV = TextureWrapMode.Clamp;
                imp.filterMode = FilterMode.Trilinear;
                imp.anisoLevel = 4;
                imp.mipmapEnabled = true;
                imp.maxTextureSize = 2048;
                imp.SaveAndReimport();
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        /// <summary>Static stadium: contributes GI and receives lightmaps (small props: light probes); big dark surfaces
        /// get fewer texels. Animated / transparent / emissive-feed parts stay out of the bake.</summary>
        static void GiFlags(GameObject stadium)
        {
            foreach (var r in stadium.GetComponentsInChildren<MeshRenderer>(true))
            {
                var go = r.gameObject;
                string n = go.name;
                // crowd cards: 4,704 thin stacked quads bake into blotches (they shade each other) → light probes
                bool skip = n.StartsWith("Flag_") || n.StartsWith("Podium_Flag") || n.StartsWith("Light_Beams") || n.StartsWith("Cauldron") || n.StartsWith("Crowd_Cards")
                            || HasParent(r.transform, "Light_Beams") || HasParent(r.transform, "ArenaLights") || r.GetComponentInParent<StadiumAtmosphere>() != null;
                var flags = GameObjectUtility.GetStaticEditorFlags(go);
                flags = skip ? flags & ~StaticEditorFlags.ContributeGI : flags | StaticEditorFlags.ContributeGI;
                GameObjectUtility.SetStaticEditorFlags(go, flags);
                if (skip)
                {
                    r.receiveGI = ReceiveGI.LightProbes;
                    continue;
                }
                var size = r.bounds.size;
                bool small = Mathf.Max(size.x, size.y, size.z) < 1.2f;
                r.receiveGI = small ? ReceiveGI.LightProbes : ReceiveGI.Lightmaps;
                var so = new SerializedObject(r);
                var scale = so.FindProperty("m_ScaleInLightmap");
                if (scale != null)
                {
                    scale.floatValue = n.StartsWith("Ceiling") || n.StartsWith("Roof") || n.StartsWith("Facade") ? 0.15f
                                     : 1f;
                    so.ApplyModifiedPropertiesWithoutUndo();
                }
            }
        }

        /// <summary>Probe grid where athletes can be: from 20 m behind the start to 110 m ahead, ±14 m across, at three
        /// heights (the scene origin is the lane-4 start, facing +x).</summary>
        static int LightProbes(GameObject stadium)
        {
            var scene = stadium.scene;
            foreach (var old in scene.GetRootGameObjects().Where(g => g.name == "ArenaProbes")) UnityEngine.Object.DestroyImmediate(old);
            var go = new GameObject("ArenaProbes");
            UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(go, scene);
            var group = go.AddComponent<LightProbeGroup>();
            var pts = new List<Vector3>();
            for (float x = -20f; x <= 110.01f; x += 6.5f)
                for (float z = -14f; z <= 14.01f; z += 4f)
                    foreach (var y in new[] { 0.3f, 1.3f, 3.0f })
                        pts.Add(new Vector3(x, y, z));
            // and a small cluster round the podium (the ceremony statues)
            var step = stadium.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == "Podium_Step_1");
            if (step != null)
                for (float dx = -6f; dx <= 6.01f; dx += 3f)
                    for (float dz = -6f; dz <= 6.01f; dz += 3f)
                        foreach (var y in new[] { 0.5f, 2f, 4f })
                            pts.Add(step.position + new Vector3(dx, y - 0.9f, dz));
            go.transform.position = Vector3.zero;
            group.probePositions = pts.ToArray();
            return pts.Count;
        }

        public static void ApplyLightingSettings(UnityEngine.SceneManagement.Scene scene)
        {
            var ls = AssetDatabase.LoadAssetAtPath<LightingSettings>(LightingPath);
            if (ls == null)
            {
                ls = new LightingSettings { name = "ArenaLighting" };
                ls.bakedGI = true;
                ls.realtimeGI = false;
                ls.lightmapper = LightingSettings.Lightmapper.ProgressiveGPU;
                ls.lightmapResolution = 3f;
                ls.lightmapPadding = 2;
                ls.lightmapMaxSize = 2048;
                ls.directSampleCount = 32;
                ls.indirectSampleCount = 256;
                ls.environmentSampleCount = 128;
                ls.maxBounces = 2;
                ls.ao = true;
                ls.aoMaxDistance = 1.5f;
                ls.aoExponentIndirect = 1f;
                ls.aoExponentDirect = 0f;
                ls.directionalityMode = LightmapsMode.NonDirectional;   // mobile: no directional lightmap
                ls.lightmapCompression = LightmapCompression.NormalQuality;
                ls.mixedBakeMode = MixedLightingMode.IndirectOnly;
                ls.filteringMode = LightingSettings.FilterMode.Auto;
                Directory.CreateDirectory(Path.GetDirectoryName(LightingPath));
                AssetDatabase.CreateAsset(ls, LightingPath);
            }
            Lightmapping.SetLightingSettingsForScene(scene, ls);
            // the arena spots are baked (athletes: light probes); the key light stays realtime for the shadows
            foreach (var root in scene.GetRootGameObjects())
                foreach (var l in root.GetComponentsInChildren<Light>(true))
                {
                    if (HasParent(l.transform, "ArenaLights")) { l.lightmapBakeType = LightmapBakeType.Baked; EditorUtility.SetDirty(l); }
                }
            EditorSceneManager.MarkSceneDirty(scene);
        }

        // ------------------------------------------------------------------------------------------------ broadcast
        /// <summary>Called by BroadcastFx.Install (inside its BroadcastFX root): the showcase part of the broadcast layer.</summary>
        public static void InstallBroadcast(BroadcastHud hud, BroadcastDirector director, GameObject root, TensionMeter tension,
                                            ArenaAudio audio, GameObject rig)
        {
            EnsureAssets();
            var scene = root.scene;
            var stadium = scene.GetRootGameObjects().FirstOrDefault(g => g.name == "Stadium");
            if (stadium != null) Dress(stadium, true);
            Transform Find(string n) => stadium == null ? null : stadium.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == n);

            // cameras stay inside the bowl: centre + long axis from the stadium's anchors
            var bowl = Find("AudioAnchor_PA_Centre");
            var s0 = Find("Speaker_00");
            var s1 = Find("Speaker_01");
            if (bowl != null && s0 != null && s1 != null)
            {
                director.bowlCentre = bowl;
                director.bowlAxis = (s1.position - s0.position).normalized;       // Speaker_00/01 sit at Blender x = ∓13 m
                EditorUtility.SetDirty(director);
            }

            // crowd
            var crowd = Child(root, "CrowdDirector").AddComponent<CrowdDirector>();
            crowd.board = hud.board;
            crowd.tension = tension;

            // stadium screens
            ScreenFeed screens = null;
            if (stadium != null)
            {
                var sgo = Child(root, "StadiumScreens");
                var doc = sgo.AddComponent<UIDocument>();
                doc.panelSettings = AssetDatabase.LoadAssetAtPath<PanelSettings>(ScreenPanel);
                screens = sgo.AddComponent<ScreenFeed>();
                screens.board = hud.board;
                screens.tension = tension;
                screens.style = AssetDatabase.LoadAssetAtPath<StyleSheet>(BoardStyle);
                screens.title = hud.title;
                screens.subtitle = hud.subtitle;
                screens.forward = director.forward;
                screens.feedTexture = AssetDatabase.LoadAssetAtPath<RenderTexture>(FeedRT);
                var cgo = Child(sgo, "ScreenFeedCam");
                var cam = cgo.AddComponent<Camera>();
                cam.targetTexture = screens.feedTexture;
                cam.fieldOfView = 38f;
                cam.nearClipPlane = 0.2f;
                cam.farClipPlane = 350f;
                cam.depth = -10;
                cam.allowHDR = false;
                cam.allowMSAA = false;
                var data = cam.GetUniversalAdditionalCameraData();
                data.renderPostProcessing = false;
                data.renderShadows = false;
                data.antialiasing = AntialiasingMode.None;
                data.requiresDepthOption = CameraOverrideOption.Off;
                data.requiresColorOption = CameraOverrideOption.Off;
                cam.enabled = false;
                screens.feedCamera = cam;
                hud.screens = screens;
            }

            // medal ceremony
            var step1 = Find("Podium_Step_1");
            if (step1 != null)
            {
                var pgo = Child(root, "PodiumCeremony");
                var pc = pgo.AddComponent<PodiumCeremony>();
                pc.board = hud.board;
                pc.raceForward = director.forward;
                pc.steps = new[] { step1, Find("Podium_Step_2"), Find("Podium_Step_3") };
                pc.flags = new[] { Find("Podium_Flag_1"), Find("Podium_Flag_2"), Find("Podium_Flag_3") };
                pc.flagTops = new[] { Find("Podium_FlagTop_1"), Find("Podium_FlagTop_2"), Find("Podium_FlagTop_3") };
                pc.podiumCam = Find("PodiumCam");
                pc.podiumLook = Find("PodiumLook");
                pc.arenaAudio = audio;
                pc.crowd = crowd;
                // statue slots (pivot › pose › 4 parts): the ceremony bakes the top three's poses into them
                var holder = Child(pgo, "Statues");
                pc.statueSlots = Enumerable.Range(1, 3).Select(k =>
                {
                    var slot = Child(holder, $"Statue_{k}");
                    var pose = Child(slot, "Pose");
                    for (int i = 0; i < 4; i++)
                    {
                        var part = Child(pose, $"Part_{i}");
                        part.AddComponent<MeshFilter>();
                        var mr = part.AddComponent<MeshRenderer>();
                        mr.shadowCastingMode = ShadowCastingMode.On;
                        mr.lightProbeUsage = LightProbeUsage.BlendProbes;
                    }
                    slot.SetActive(false);
                    return slot.transform;
                }).ToArray();
                var conf = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(ConfettiPrefab), pgo.transform);
                conf.transform.position = pc.podiumLook.position + Vector3.up * 1.5f;
                pc.confetti = conf.GetComponent<ParticleSystem>();
                // fixed podium camera in the Cinemachine rig
                var cmgo = Child(rig, "CM_Podium");
                cmgo.transform.position = pc.podiumCam.position;
                cmgo.transform.LookAt(pc.podiumLook);
                var cm = cmgo.AddComponent<CinemachineCamera>();
                var lens = LensSettings.FromCamera(director.GetComponent<Camera>());
                lens.FieldOfView = 46f;
                cm.Lens = lens;
                cm.LookAt = pc.podiumLook;
                cm.Priority = 10;
                // the result card covers the upper-middle of the view: frame the podium low (CM3: +y = lower)
                var aim = cmgo.AddComponent<CinemachineRotationComposer>();
                var comp = aim.Composition;
                comp.ScreenPosition = new Vector2(0f, 0.24f);
                aim.Composition = comp;
                director.podium = cm;
                director.ceremony = pc;
                EditorUtility.SetDirty(director);
            }

            // audio: mixer groups + snapshots, crowd sectors, PA, event SFX, reverb, hit-stop filter
            var mixer = EnsureMixer();
            AudioMixerGroup G(string n) => mixer != null ? mixer.FindMatchingGroups(n).FirstOrDefault(g => g.name == n) : null;
            AudioMixerSnapshot S(string n) => mixer != null ? mixer.FindSnapshot(n) : null;
            audio.mixer = mixer;
            audio.snapReady = S("Ready"); audio.snapLive = S("Live"); audio.snapResult = S("Result"); audio.snapAnnounce = S("Announce");
            foreach (var s in new[] { audio.bed, audio.tensionLayer, audio.stinger }) if (s != null) s.outputAudioMixerGroup = G("Crowd");
            if (audio.ui != null) audio.ui.outputAudioMixerGroup = G("Ui");
            foreach (var v in audio.voices) v.outputAudioMixerGroup = G("Sfx");
            var ago = audio.gameObject;
            var bedClip = audio.bed != null ? audio.bed.clip : null;
            var sectors = new List<AudioSource>();
            foreach (var dir in new[] { "S", "N", "E", "W" })
            {
                var a = Find("AudioAnchor_Crowd_" + dir);
                if (a == null) continue;
                var s = Source(ago, "CrowdSector_" + dir, bedClip, true, 0.3f, G("Crowd"));
                s.transform.position = a.position;
                s.spatialBlend = 0.85f;
                s.rolloffMode = AudioRolloffMode.Logarithmic;
                s.minDistance = 30f;
                s.maxDistance = 260f;
                s.dopplerLevel = 0f;
                s.spread = 120f;
                sectors.Add(s);
            }
            audio.crowdSectors = sectors.ToArray();
            var paAnchor = Find("AudioAnchor_PA_Centre");
            var pa = Source(ago, "PA_Announcer", null, false, 1f, G("Announcer"));
            if (paAnchor != null) pa.transform.position = paAnchor.position;
            pa.spatialBlend = 0.35f;
            pa.dopplerLevel = 0f;
            pa.minDistance = 40f;
            pa.maxDistance = 300f;
            var echo = pa.gameObject.AddComponent<AudioEchoFilter>();
            echo.delay = 130f;
            echo.decayRatio = 0.3f;
            echo.wetMix = 0.35f;
            echo.dryMix = 1f;
            audio.pa = pa;
            audio.eventSfx = Source(ago, "EventSfx", null, false, 1f, G("Sfx"));
            if (paAnchor != null)
            {
                var rz = Child(ago, "ArenaReverb").AddComponent<AudioReverbZone>();
                rz.transform.position = paAnchor.position;
                rz.reverbPreset = AudioReverbPreset.Arena;
                rz.minDistance = 160f;
                rz.maxDistance = 260f;
            }
            var listenerGo = director.gameObject;
            var lp = listenerGo.GetComponent<AudioLowPassFilter>();     // (no ??: GetComponent returns a fake null in the editor)
            if (lp == null) lp = listenerGo.AddComponent<AudioLowPassFilter>();
            lp.cutoffFrequency = 22000f;
            lp.lowpassResonanceQ = 1f;
            lp.enabled = false;
            audio.listenerLowPass = lp;
            audio.crowd = crowd;
            string ev = BroadcastFx.AudioFolder + "/Sfx/Event/", an = BroadcastFx.AudioFolder + "/Announcer/";
            AudioClip C(string p) => AssetDatabase.LoadAssetAtPath<AudioClip>(p);
            audio.starterGun = C(ev + "starter_gun.wav"); audio.whistle = C(ev + "whistle.wav"); audio.airHorn = C(ev + "air_horn.wav");
            audio.buzzer = C(ev + "buzzer.wav"); audio.paChime = C(ev + "pa_chime.wav"); audio.fanfare = C(ev + "fanfare.wav");
            audio.cameraFlash = C(ev + "camera_flash.wav");
            audio.winLines = Enumerable.Range(1, 8).Select(i => C($"{an}win_L{i}.wav")).ToArray();
            audio.outLines = Enumerable.Range(1, 8).Select(i => C($"{an}out_L{i}.wav")).ToArray();
            audio.leadLines = Enumerable.Range(1, 8).Select(i => C($"{an}lead_L{i}.wav")).ToArray();
            audio.recordLine = C(an + "record.wav"); audio.marksLine = C(an + "marks.wav"); audio.setLine = C(an + "set.wav");
            audio.saveLine = C(an + "save.wav"); audio.nextHeatLine = C(an + "next_heat.wav"); audio.resultLine = C(an + "result.wav");
            audio.photoFinishLine = C(an + "photo_finish.wav");
            EditorUtility.SetDirty(audio);

            // performance overlay
            var perfGo = Child(root, "PerfOverlay");
            var pdoc = perfGo.AddComponent<UIDocument>();
            pdoc.panelSettings = hud.GetComponent<UIDocument>().panelSettings;
            pdoc.sortingOrder = 10;
            var perf = perfGo.AddComponent<PerfOverlay>();
            perf.style = AssetDatabase.LoadAssetAtPath<StyleSheet>(PerfStyle);
            perf.screens = screens;
            PlayerSettings.enableFrameTimingStats = true;
            EditorUtility.SetDirty(hud);
        }

        static GameObject Child(GameObject parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            return go;
        }

        static AudioSource Source(GameObject parent, string name, AudioClip clip, bool loop, float volume, AudioMixerGroup group)
        {
            var s = Child(parent, name).AddComponent<AudioSource>();
            s.clip = clip;
            s.loop = loop;
            s.playOnAwake = false;
            s.volume = volume;
            s.spatialBlend = 0f;
            s.outputAudioMixerGroup = group;
            return s;
        }

        // ------------------------------------------------------------------------------------------------ assets
        public static void EnsureAssets()
        {
            Directory.CreateDirectory(Mats);
            var board = RT(BoardRT, 1280, 448);
            RT(FeedRT, 560, 300);
            ScreenMaterial("ScreenLive_Wide", board, Vector2.one, Vector2.zero);
            ScreenMaterial("ScreenLive_Cube", board, new Vector2(CubeCropU, 1f), new Vector2((1f - CubeCropU) / 2f, 0f));
            var beamPath = Mats + "/LightBeam.mat";
            if (AssetDatabase.LoadAssetAtPath<Material>(beamPath) == null)
                AssetDatabase.CreateAsset(new Material(Shader.Find("PoOlympic/LightBeam") ?? throw new InvalidOperationException("PoOlympic/LightBeam missing")), beamPath);
            if (AssetDatabase.LoadAssetAtPath<PanelSettings>(ScreenPanel) == null)
            {
                var ps = ScriptableObject.CreateInstance<PanelSettings>();
                ps.targetTexture = board;
                ps.scaleMode = PanelScaleMode.ConstantPixelSize;
                ps.scale = 1f;
                ps.clearColor = true;
                ps.colorClearValue = new Color(0.02f, 0.04f, 0.09f, 1f);
                ps.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(Theme);
                ps.sortingOrder = -10;
                AssetDatabase.CreateAsset(ps, ScreenPanel);
            }
            EnsureConfetti();
            EnsureMixer();
            AudioImport();
            AssetDatabase.SaveAssets();
        }

        static RenderTexture RT(string path, int w, int h)
        {
            var rt = AssetDatabase.LoadAssetAtPath<RenderTexture>(path);
            if (rt != null) return rt;
            rt = new RenderTexture(w, h, 16, RenderTextureFormat.ARGB32) { name = Path.GetFileNameWithoutExtension(path), useMipMap = true, autoGenerateMips = true, filterMode = FilterMode.Trilinear, anisoLevel = 4 };
            AssetDatabase.CreateAsset(rt, path);
            return rt;
        }

        static void ScreenMaterial(string name, Texture tex, Vector2 tiling, Vector2 offset)
        {
            var path = $"{Mats}/{name}.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m == null)
            {
                m = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
                AssetDatabase.CreateAsset(m, path);
            }
            m.SetTexture("_BaseMap", tex);
            m.SetTextureScale("_BaseMap", tiling);
            m.SetTextureOffset("_BaseMap", offset);
            m.SetColor("_BaseColor", new Color(1.5f, 1.5f, 1.5f, 1f));     // a lit screen: over bloom threshold on white
            EditorUtility.SetDirty(m);
        }

        static void EnsureConfetti()
        {
            if (AssetDatabase.LoadAssetAtPath<GameObject>(ConfettiPrefab) != null) return;
            var matPath = BroadcastFx.FxFolder + "/FxConfetti.mat";
            var mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
            if (mat == null)
            {
                mat = new Material(Shader.Find("Universal Render Pipeline/Particles/Unlit")) { name = "FxConfetti" };
                mat.SetFloat("_Cull", 0f);
                AssetDatabase.CreateAsset(mat, matPath);
            }
            var go = new GameObject("Confetti");
            try
            {
                var ps = go.AddComponent<ParticleSystem>();
                ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
                var main = ps.main;
                main.duration = 1f; main.loop = false; main.playOnAwake = false;
                main.startLifetime = new ParticleSystem.MinMaxCurve(4f, 6.5f);
                main.startSpeed = new ParticleSystem.MinMaxCurve(6f, 12f);
                main.startSize3D = true;
                main.startSizeX = new ParticleSystem.MinMaxCurve(0.07f, 0.12f);
                main.startSizeY = new ParticleSystem.MinMaxCurve(0.04f, 0.07f);
                main.startSizeZ = 1f;
                main.startRotation3D = true;
                main.startRotationX = new ParticleSystem.MinMaxCurve(0f, Mathf.PI * 2f);
                main.startRotationY = new ParticleSystem.MinMaxCurve(0f, Mathf.PI * 2f);
                main.startRotationZ = new ParticleSystem.MinMaxCurve(0f, Mathf.PI * 2f);
                var grad = new Gradient();
                grad.SetKeys(new[] { new GradientColorKey(new Color(1f, 0.82f, 0.2f), 0f), new GradientColorKey(new Color(0.2f, 0.55f, 1f), 0.25f),
                                     new GradientColorKey(new Color(1f, 0.3f, 0.35f), 0.5f), new GradientColorKey(new Color(0.3f, 0.9f, 0.5f), 0.75f),
                                     new GradientColorKey(Color.white, 1f) },
                             new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(1f, 1f) });
                main.startColor = new ParticleSystem.MinMaxGradient(grad) { mode = ParticleSystemGradientMode.RandomColor };
                main.gravityModifier = 0.35f;
                main.maxParticles = 500;
                main.simulationSpace = ParticleSystemSimulationSpace.World;
                main.useUnscaledTime = true;
                var em = ps.emission; em.rateOverTime = 0f; em.SetBursts(new[] { new ParticleSystem.Burst(0f, 260), new ParticleSystem.Burst(0.35f, 160) });
                var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Cone; sh.angle = 30f; sh.radius = 1.5f; sh.rotation = new Vector3(-90f, 0f, 0f);
                var drag = ps.limitVelocityOverLifetime; drag.enabled = true; drag.dampen = 0.08f; drag.limit = 1.2f;
                var rot = ps.rotationOverLifetime; rot.enabled = true; rot.separateAxes = true;
                rot.x = new ParticleSystem.MinMaxCurve(-6f, 6f); rot.y = new ParticleSystem.MinMaxCurve(-6f, 6f); rot.z = new ParticleSystem.MinMaxCurve(-6f, 6f);
                var noise = ps.noise; noise.enabled = true; noise.strength = 0.6f; noise.frequency = 0.4f;
                var r = go.GetComponent<ParticleSystemRenderer>();
                r.renderMode = ParticleSystemRenderMode.Mesh;
                r.mesh = Resources.GetBuiltinResource<Mesh>("Quad.fbx");
                r.alignment = ParticleSystemRenderSpace.Local;
                r.sharedMaterial = mat;
                r.shadowCastingMode = ShadowCastingMode.Off;
                PrefabUtility.SaveAsPrefabAsset(go, ConfettiPrefab);
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        /// <summary>Audio/PoOlympicMix.mixer: Master › Crowd, Sfx, Ui, Announcer; snapshots Ready (hush), Live, Result (swell),
        /// Announce (crowd −9 dB, SFX −5 dB: ducking under the PA). Built through the editor's AudioMixerController (no
        /// public API creates mixers); kept if present so hand tuning survives.</summary>
        static AudioMixer EnsureMixer()
        {
            var existing = AssetDatabase.LoadAssetAtPath<AudioMixer>(MixerPath);
            if (existing != null) return existing;
            try
            {
                var asm = typeof(UnityEditor.Editor).Assembly;
                var ctrlT = asm.GetType("UnityEditor.Audio.AudioMixerController");
                var groupT = asm.GetType("UnityEditor.Audio.AudioMixerGroupController");
                const BindingFlags F = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
                var ctrl = ctrlT.GetMethod("CreateMixerControllerAtPath", F).Invoke(null, new object[] { MixerPath });
                var master = ctrlT.GetProperty("masterGroup", F).GetValue(ctrl);
                var groups = new Dictionary<string, object>();
                foreach (var n in new[] { "Crowd", "Sfx", "Ui", "Announcer" })
                {
                    var g = ctrlT.GetMethod("CreateNewGroup", F).Invoke(ctrl, new object[] { n, false });
                    ctrlT.GetMethod("AddChildToParent", F).Invoke(ctrl, new[] { g, master });
                    groups[n] = g;           // (AddGroupToCurrentView throws on a fresh mixer: it has no views yet)
                }
                var snapsProp = ctrlT.GetProperty("snapshots", F);
                var first = ((Array)snapsProp.GetValue(ctrl)).GetValue(0);
                ((UnityEngine.Object)first).name = "Ready";
                foreach (var n in new[] { "Live", "Result", "Announce" })
                {
                    ctrlT.GetMethod("CloneNewSnapshotFromTarget", F).Invoke(ctrl, new object[] { false });
                    var arr = (Array)snapsProp.GetValue(ctrl);
                    ((UnityEngine.Object)arr.GetValue(arr.Length - 1)).name = n;
                }
                var levels = new Dictionary<string, (float crowd, float sfx, float ui, float pa)>
                {
                    ["Ready"] = (-5f, 0f, 0f, 0f), ["Live"] = (0f, 0f, 0f, 0f), ["Result"] = (2f, 0f, 0f, 0f), ["Announce"] = (-9f, -5f, -3f, 1f),
                };
                var setVol = groupT.GetMethod("SetValueForVolume", F);
                foreach (var snap in (Array)snapsProp.GetValue(ctrl))
                {
                    var name = ((UnityEngine.Object)snap).name;
                    if (!levels.TryGetValue(name, out var lv)) continue;
                    setVol.Invoke(groups["Crowd"], new[] { ctrl, snap, (object)lv.crowd });
                    setVol.Invoke(groups["Sfx"], new[] { ctrl, snap, (object)lv.sfx });
                    setVol.Invoke(groups["Ui"], new[] { ctrl, snap, (object)lv.ui });
                    setVol.Invoke(groups["Announcer"], new[] { ctrl, snap, (object)lv.pa });
                }
                EditorUtility.SetDirty((UnityEngine.Object)ctrl);
                AssetDatabase.SaveAssets();
                return AssetDatabase.LoadAssetAtPath<AudioMixer>(MixerPath);
            }
            catch (Exception e)
            {
                Debug.LogWarning("[StadiumShowcase] could not build the audio mixer (plain sources are used): " + e);
                return null;
            }
        }

        /// <summary>Announcer lines + event SFX: mono, Vorbis, decompressed on load (short clips); the fanfare streams
        /// compressed.</summary>
        static void AudioImport()
        {
            foreach (var folder in new[] { BroadcastFx.AudioFolder + "/Announcer", BroadcastFx.AudioFolder + "/Sfx/Event" })
            {
                if (!AssetDatabase.IsValidFolder(folder)) continue;
                foreach (var guid in AssetDatabase.FindAssets("t:AudioClip", new[] { folder }))
                {
                    var path = AssetDatabase.GUIDToAssetPath(guid);
                    var imp = (AudioImporter)AssetImporter.GetAtPath(path);
                    var s = imp.defaultSampleSettings;
                    bool longClip = path.EndsWith("fanfare.wav");
                    var load = longClip ? AudioClipLoadType.CompressedInMemory : AudioClipLoadType.DecompressOnLoad;
                    if (imp.forceToMono && s.loadType == load && s.compressionFormat == AudioCompressionFormat.Vorbis) continue;
                    s.loadType = load;
                    s.compressionFormat = AudioCompressionFormat.Vorbis;
                    s.quality = 0.7f;
                    imp.forceToMono = true;
                    imp.defaultSampleSettings = s;
                    imp.SaveAndReimport();
                }
            }
        }
    }
}
