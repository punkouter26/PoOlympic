using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Unity half of the stadium realism pass (Blender half: SourceArt/Stadium/build_realism.py). Builds, once, the
    /// assets the user tunes by hand afterwards:
    ///   Settings/StadiumLook_Volume.asset   ACES tonemapping, bloom, colour adjustments, vignette
    ///   Materials/Sky_Stadium.mat           procedural sky
    ///   Prefabs/StadiumAtmosphere.prefab    StadiumAtmosphere (sky/fog/ambient) + global Volume + reflection probe +
    ///                                       4 floodlight banks on the roof's floodlight ring
    /// and dresses every placed stadium (EventScenes.PlaceStadium): atmosphere instance, flag wave, cauldron flicker,
    /// camera post-processing. Existing assets are kept, so Inspector tweaks survive scene rebuilds; use
    /// "Rebuild Stadium Look assets" to regenerate them. All render-only.
    /// </summary>
    public static class StadiumLook
    {
        const string ProfilePath = "Assets/PoOlympic/Settings/StadiumLook_Volume.asset";
        const string SkyPath = "Assets/PoOlympic/Materials/Sky_Stadium.mat";
        const string PrefabPath = "Assets/PoOlympic/Prefabs/StadiumAtmosphere.prefab";

        [MenuItem("PoOlympic/Stadium/Rebuild Stadium Look assets")]
        public static void RebuildAssets()
        {
            foreach (var p in new[] { PrefabPath, ProfilePath, SkyPath })
                AssetDatabase.DeleteAsset(p);
            EnsurePrefab();
            Debug.Log($"[StadiumLook] rebuilt {PrefabPath}, {ProfilePath}, {SkyPath}");
        }

        /// <summary>Apply the current code defaults to the EXISTING assets in place (keeps their GUIDs, so built scenes
        /// pick the change up without a rebuild).</summary>
        [MenuItem("PoOlympic/Stadium/Retune Stadium Look assets in place")]
        public static string Retune()
        {
            var p = AssetDatabase.LoadAssetAtPath<VolumeProfile>(ProfilePath);
            foreach (var c in p.components.ToArray()) { p.Remove(c.GetType()); UnityEngine.Object.DestroyImmediate(c, true); }
            FillProfile(p);
            var root = PrefabUtility.LoadPrefabContents(PrefabPath);
            try
            {
                UnityEngine.Object.DestroyImmediate(root.transform.Find("Floodlights").gameObject);
                AddFloodlights(root.transform);
                PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            }
            finally { PrefabUtility.UnloadPrefabContents(root); }
            AssetDatabase.SaveAssets();
            return "retuned " + ProfilePath + ", " + PrefabPath;
        }

        /// <summary>Called by EventScenes.PlaceStadium on the fresh stadium instance.</summary>
        public static void Dress(GameObject stadium)
        {
            var prefab = EnsurePrefab();
            var atmo = (GameObject)PrefabUtility.InstantiatePrefab(prefab, stadium.scene);
            atmo.transform.SetParent(stadium.transform, false);
            atmo.GetComponent<StadiumAtmosphere>().Apply();

            foreach (var t in stadium.GetComponentsInChildren<Transform>(true))
            {
                if (t.name.StartsWith("Flag_") && t.GetComponent<FlagWave>() == null)
                    t.gameObject.AddComponent<FlagWave>();
                else if (t.name == "Cauldron_Flame" && t.GetComponent<FlameFlicker>() == null)
                {
                    var f = t.gameObject.AddComponent<FlameFlicker>();
                    var glow = new GameObject("FlameGlow").AddComponent<Light>();
                    glow.transform.SetParent(t, false);
                    glow.transform.localPosition = new Vector3(0f, 3f, 0f);
                    glow.type = LightType.Point;
                    glow.color = new Color(1f, 0.62f, 0.25f);
                    glow.range = 45f;
                    glow.shadows = LightShadows.None;
                    f.glow = glow;
                }
            }

            foreach (var root in stadium.scene.GetRootGameObjects())
                foreach (var cam in root.GetComponentsInChildren<Camera>(true))
                {
                    var data = cam.GetUniversalAdditionalCameraData();
                    data.renderPostProcessing = true;
                    data.antialiasing = AntialiasingMode.FastApproximateAntialiasing;
                }
            ArenaLighting(stadium);
            StadiumShowcase.Dress(stadium, false);        // screens go live when the broadcast layer is installed
        }

        // ---------------------------------------------------------------- indoor arena (SourceArt/Stadium/build_indoor.py)
        public const float SpotIntensity = 450f, WashIntensity = 420f;   // wash baked since the showcase pass (free at runtime)
        public const int FieldSpots = 24;                       // Spot_00-23 field rig, Spot_24+ crowd wash

        /// <summary>
        /// Indoor arena lighting on a placed stadium (user decision 2026-09-28: closed roof + spotlights; the open bowl put
        /// the home straight in the canopy's shadow): nothing above 25 m casts shadows (ceiling, trusses, fixtures, roof), the
        /// scene's directional light becomes a near-vertical shadow-casting key light, one spot light per Spot_## fixture
        /// aimed at its SpotAim_## (no shadows; created as scene objects under Stadium/ArenaLights so they can be tuned in
        /// the scene), the roof-ring floodlights of the open-air look are switched off, the atmosphere goes indoor
        /// (trilight ambient), and the static stadium is marked for static batching (flags + flame stay dynamic).
        /// </summary>
        public static int ArenaLighting(GameObject stadium)
        {
            float ground = stadium.transform.position.y;
            foreach (var r in stadium.GetComponentsInChildren<Renderer>(true))
                if (r.bounds.min.y - ground > 25f) r.shadowCastingMode = ShadowCastingMode.Off;

            foreach (var root in stadium.scene.GetRootGameObjects())
                foreach (var l in root.GetComponentsInChildren<Light>(true))
                    if (l.type == LightType.Directional)
                    {
                        l.transform.rotation = Quaternion.Euler(80f, 30f, 0f);   // near-vertical: stands cast no shade on the track
                        l.intensity = 1.1f;
                        l.color = new Color(1f, 0.97f, 0.93f);
                        l.shadows = LightShadows.Soft;
                        l.shadowStrength = 0.7f;
                        EditorUtility.SetDirty(l);
                    }

            var old = stadium.transform.Find("ArenaLights");
            if (old != null) UnityEngine.Object.DestroyImmediate(old.gameObject);
            var rig = new GameObject("ArenaLights").transform;
            rig.SetParent(stadium.transform, false);
            var all = stadium.GetComponentsInChildren<Transform>(true);
            int n = 0;
            foreach (var t in all.Where(x => System.Text.RegularExpressions.Regex.IsMatch(x.name, @"^Spot_\d\d$")).OrderBy(x => x.name))
            {
                var aim = all.FirstOrDefault(x => x.name == "SpotAim_" + t.name.Substring(5));
                if (aim == null) continue;
                int k = int.Parse(t.name.Substring(5));
                bool wash = k >= FieldSpots;
                var l = new GameObject($"ArenaSpot_{k:00}").AddComponent<Light>();
                l.transform.SetParent(rig, false);
                l.transform.SetPositionAndRotation(t.position, Quaternion.LookRotation(aim.position - t.position));
                l.type = LightType.Spot;
                // crowd wash: wide + very soft (showcase pass: the 80/50 cones left 12 hard ovals on the stands)
                l.spotAngle = wash ? 115f : 70f;
                l.innerSpotAngle = wash ? 30f : 40f;
                l.range = Vector3.Distance(t.position, aim.position) * 1.8f;
                l.intensity = wash ? WashIntensity : SpotIntensity;
                l.color = new Color(1f, 0.97f, 0.92f);
                l.shadows = LightShadows.None;
                n++;
            }

            var atmo = stadium.GetComponentInChildren<StadiumAtmosphere>(true);
            if (atmo != null)
            {
                var flood = atmo.transform.Find("Floodlights");
                if (flood != null) flood.gameObject.SetActive(false);
                atmo.indoor = true;
                atmo.Apply();
                EditorUtility.SetDirty(atmo);
            }

            foreach (var t in stadium.GetComponentsInChildren<Transform>(true))
            {
                bool animated = t.name.StartsWith("Flag_") || t.name == "Cauldron_Flame" || t.GetComponentInParent<Light>() != null
                                || t.GetComponent<StadiumAtmosphere>() != null || t.name == "ArenaLights";
                GameObjectUtility.SetStaticEditorFlags(t.gameObject, animated ? 0 :
                    StaticEditorFlags.BatchingStatic | StaticEditorFlags.OccludeeStatic | StaticEditorFlags.ReflectionProbeStatic);
            }
            return n;
        }

        /// <summary>Re-light every built scene that holds a placed stadium (indoor arena), without rebuilding it.</summary>
        [MenuItem("PoOlympic/Stadium/Apply indoor arena lighting to all scenes")]
        public static string RelightAllScenes()
        {
            var report = new System.Text.StringBuilder();
            foreach (var guid in AssetDatabase.FindAssets("t:Scene", new[] { "Assets/PoOlympic/Scenes" }))
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var scene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene(path, UnityEditor.SceneManagement.OpenSceneMode.Single);
                var stadium = scene.GetRootGameObjects().FirstOrDefault(g => g.name == "Stadium");
                if (stadium == null) continue;
                int n = ArenaLighting(stadium);
                UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene);
                report.Append($"{Path.GetFileNameWithoutExtension(path)}: {n} spots; ");
            }
            return report.ToString();
        }

        static GameObject EnsurePrefab()
        {
            var existing = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
            if (existing != null) return existing;
            Directory.CreateDirectory(Path.GetDirectoryName(PrefabPath));

            var root = new GameObject("StadiumAtmosphere");
            try
            {
                var atmo = root.AddComponent<StadiumAtmosphere>();
                atmo.skybox = EnsureSky();

                var vol = new GameObject("PostVolume").AddComponent<Volume>();
                vol.transform.SetParent(root.transform, false);
                vol.isGlobal = true;
                vol.sharedProfile = EnsureProfile();

                // one realtime capture at scene start (cheap on phones, no bake step), covering the bowl
                var probe = new GameObject("ReflectionProbe").AddComponent<ReflectionProbe>();
                probe.transform.SetParent(root.transform, false);
                probe.transform.localPosition = new Vector3(0f, 4f, 0f);
                probe.mode = ReflectionProbeMode.Realtime;
                probe.refreshMode = ReflectionProbeRefreshMode.OnAwake;
                probe.timeSlicingMode = ReflectionProbeTimeSlicingMode.NoTimeSlicing;
                probe.size = new Vector3(260f, 80f, 200f);
                probe.resolution = 128;
                probe.hdr = true;

                AddFloodlights(root.transform);
                return PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            }
            finally { UnityEngine.Object.DestroyImmediate(root); }
        }

        /// <summary>Four floodlight banks at the diagonal extremes of the roof's Floodlight_Ring (stadium-local).</summary>
        static void AddFloodlights(Transform parent)
        {
            var stadium = AssetDatabase.LoadAssetAtPath<GameObject>(EventScenes.StadiumAsset);
            var ring = stadium.GetComponentsInChildren<MeshFilter>(true).FirstOrDefault(m => m.name.StartsWith("Floodlight_Ring"))
                       ?? throw new MissingReferenceException("Floodlight_Ring not in " + EventScenes.StadiumAsset);
            var verts = ring.sharedMesh.vertices.Select(v => stadium.transform.InverseTransformPoint(ring.transform.TransformPoint(v))).ToArray();
            var banks = new GameObject("Floodlights").transform;
            banks.SetParent(parent, false);
            int k = 0;
            foreach (var (sx, sz) in new[] { (1f, 1f), (-1f, 1f), (-1f, -1f), (1f, -1f) })
            {
                var dir = new Vector2(sx, sz).normalized;
                var p = verts.OrderByDescending(v => v.x * dir.x + v.z * dir.y).First();
                var l = new GameObject($"Floodlight_{k++}").AddComponent<Light>();
                l.transform.SetParent(banks, false);
                // 3 m in from the ring and 2 m below it: a light ON the roof geometry blows the roof up to thousands
                // of times the scene brightness and bloom smears it over the whole frame
                var inward = new Vector3(-p.x, 0f, -p.z).normalized;
                p += inward * 3f + Vector3.down * 2f;
                l.transform.localPosition = p;
                l.transform.localRotation = Quaternion.LookRotation(new Vector3(0f, 0f, 0f) - p);
                l.type = LightType.Spot;
                l.spotAngle = 75f;
                l.innerSpotAngle = 45f;
                l.range = 180f;
                l.intensity = 150f;
                l.color = new Color(1f, 0.97f, 0.92f);
                l.shadows = LightShadows.None;
            }
        }

        static VolumeProfile EnsureProfile()
        {
            var p = AssetDatabase.LoadAssetAtPath<VolumeProfile>(ProfilePath);
            if (p != null) return p;
            Directory.CreateDirectory(Path.GetDirectoryName(ProfilePath));
            p = ScriptableObject.CreateInstance<VolumeProfile>();
            AssetDatabase.CreateAsset(p, ProfilePath);
            FillProfile(p);
            return p;
        }

        static void FillProfile(VolumeProfile p)
        {
            var tm = Add<Tonemapping>(p);
            tm.mode.value = TonemappingMode.Neutral;   // ACES crushed the dark track red to near black
            var bloom = Add<Bloom>(p);
            bloom.threshold.value = 1.1f;
            bloom.intensity.value = 0.2f;
            bloom.scatter.value = 0.55f;
            bloom.clamp.value = 8f;                     // no single hot pixel can flood the frame
            var ca = Add<ColorAdjustments>(p);
            ca.postExposure.value = 0.35f;
            ca.contrast.value = 12f;
            ca.saturation.value = 8f;
            var vig = Add<Vignette>(p);
            vig.intensity.value = 0.18f;
            vig.smoothness.value = 0.4f;

            EditorUtility.SetDirty(p);
            AssetDatabase.SaveAssets();
        }

        static T Add<T>(VolumeProfile p) where T : VolumeComponent
        {
            var c = p.Add<T>(true);
            c.name = typeof(T).Name;
            AssetDatabase.AddObjectToAsset(c, p);
            return c;
        }

        static Material EnsureSky()
        {
            var m = AssetDatabase.LoadAssetAtPath<Material>(SkyPath);
            if (m != null) return m;
            m = new Material(Shader.Find("Skybox/Procedural") ?? throw new InvalidOperationException("Skybox/Procedural shader missing"));
            m.SetFloat("_SunSize", 0.035f);
            m.SetFloat("_SunSizeConvergence", 6f);
            m.SetFloat("_AtmosphereThickness", 0.85f);
            m.SetColor("_SkyTint", new Color(0.52f, 0.6f, 0.72f));
            m.SetColor("_GroundColor", new Color(0.33f, 0.35f, 0.33f));
            m.SetFloat("_Exposure", 1.15f);
            AssetDatabase.CreateAsset(m, SkyPath);
            return m;
        }
    }
}
