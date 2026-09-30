using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Showcase pass (GFX / sound top 10): the exported stadium, the shaders and the authored event scenes.</summary>
    public class ShowcaseTests
    {
        const string Stadium = "Assets/PoOlympic/Art/Stadium/Stadium.glb";

        [Test]
        public void StadiumHasTheShowcaseAnchorsAndNoOlympicMarks()
        {
            var st = AssetDatabase.LoadAssetAtPath<GameObject>(Stadium);
            Assert.IsNotNull(st, Stadium);
            var names = st.GetComponentsInChildren<Transform>(true).Select(t => t.name).ToHashSet();
            foreach (var n in new[] { "VideoCube", "Crowd_Cards", "Light_Beams", "Roof_Rigging", "Podium", "PodiumCam", "PodiumLook",
                                      "Podium_Step_1", "Podium_Step_2", "Podium_Step_3", "Podium_Flag_1", "Podium_FlagTop_3",
                                      "AudioAnchor_PA_Centre", "AudioAnchor_Crowd_N", "AudioAnchor_Crowd_S", "AudioAnchor_Crowd_E", "AudioAnchor_Crowd_W",
                                      "Speaker_00", "Speaker_07", "E01_L3", "E19_L7" })
                Assert.IsTrue(names.Contains(n), $"{Stadium}: {n} missing");
            var mats = st.GetComponentsInChildren<Renderer>(true).SelectMany(r => r.sharedMaterials).Where(m => m != null).Select(m => m.name).ToList();
            Assert.IsFalse(mats.Any(m => m.StartsWith("Rings_")), "Olympic ring materials are exported (docs/LICENSING.md)");
            Assert.IsFalse(names.Contains("Cauldron_Flame") || names.Contains("Identity"), "exterior Olympic identity is exported");
            Assert.IsEmpty(st.GetComponentsInChildren<Collider>(true), "stadium must stay render-only");
        }

        [Test]
        public void ShowcaseShadersCompile()
        {
            foreach (var n in new[] { "PoOlympic/Crowd", "PoOlympic/LightBeam" })
            {
                var s = Shader.Find(n);
                Assert.IsNotNull(s, n);
                Assert.IsFalse(ShaderUtil.ShaderHasError(s), n + ": " + string.Join(" | ", ShaderUtil.GetShaderMessages(s).Select(m => m.message)));
            }
        }

        [Test]
        public void EventScenesCarryTheShowcase()
        {
            var current = EditorSceneManager.GetActiveScene().path;
            try
            {
                foreach (var path in Editor.EventScenes.EventScenePaths.Values.Distinct())
                {
                    EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                    Assert.IsNotNull(Object.FindAnyObjectByType<CrowdDirector>(FindObjectsInactive.Include), $"{path}: CrowdDirector");
                    var screens = Object.FindAnyObjectByType<ScreenFeed>(FindObjectsInactive.Include);
                    Assert.IsTrue(screens != null && screens.feedCamera != null && screens.feedTexture != null, $"{path}: ScreenFeed");
                    var hud = Object.FindAnyObjectByType<BroadcastHud>(FindObjectsInactive.Include);
                    Assert.AreSame(screens, hud.screens, $"{path}: HUD → screens");
                    var pc = Object.FindAnyObjectByType<PodiumCeremony>(FindObjectsInactive.Include);
                    Assert.IsTrue(pc != null && pc.steps.All(s => s != null) && pc.flags.All(f => f != null) && pc.podiumCam != null, $"{path}: PodiumCeremony");
                    var dir = Object.FindAnyObjectByType<BroadcastDirector>(FindObjectsInactive.Include);
                    Assert.IsTrue(dir.podium != null && dir.ceremony == pc, $"{path}: podium camera");
                    var audio = Object.FindAnyObjectByType<ArenaAudio>(FindObjectsInactive.Include);
                    Assert.IsTrue(audio.mixer != null && audio.snapLive != null && audio.snapAnnounce != null, $"{path}: mixer + snapshots");
                    Assert.AreEqual(4, audio.crowdSectors.Length, $"{path}: crowd sectors");
                    Assert.IsTrue(audio.pa != null && audio.starterGun != null && audio.whistle != null && audio.fanfare != null, $"{path}: PA + event SFX");
                    Assert.IsTrue(audio.winLines.All(c => c != null) && audio.outLines.All(c => c != null) && audio.leadLines.All(c => c != null), $"{path}: announcer lines");
                    Assert.IsNotNull(Object.FindAnyObjectByType<PerfOverlay>(FindObjectsInactive.Include), $"{path}: PerfOverlay");
                    Assert.IsNotNull(Object.FindAnyObjectByType<LightProbeGroup>(FindObjectsInactive.Include), $"{path}: light probes");
                    var crowd = Object.FindObjectsByType<Renderer>(FindObjectsInactive.Include).SelectMany(r => r.sharedMaterials).Count(m => m != null && m.shader.name == "PoOlympic/Crowd");
                    Assert.Greater(crowd, 0, $"{path}: living crowd material");
                    Assert.Greater(LightmapSettings.lightmaps.Length, 0, $"{path}: baked lightmap");
                }
            }
            finally
            {
                if (!string.IsNullOrEmpty(current) && File.Exists(current)) EditorSceneManager.OpenScene(current, OpenSceneMode.Single);
            }
        }
    }
}
