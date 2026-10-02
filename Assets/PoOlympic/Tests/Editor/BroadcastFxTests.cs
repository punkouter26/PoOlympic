using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Broadcast features 2/3/5/6/7/10 + world records: the pure logic and the authored event scenes.</summary>
    public class BroadcastFxTests
    {
        const int TestEvent = 99;          // not a real event: keeps the device's records untouched

        [TearDown]
        public void Clean() => Records.Clear(TestEvent);

        [Test]
        public void RecordsKeepTopFiveBestFirst()
        {
            Records.Clear(TestEvent);
            Assert.AreEqual(1, Records.Submit(TestEvent, 9.5, true, "9.50 s", "L1", "MATT"));
            Assert.AreEqual(1, Records.Submit(TestEvent, 9.1, true, "9.10 s", "Z2", "ZOMBIE"), "faster = new world record");
            Assert.AreEqual(2, Records.Submit(TestEvent, 9.3, true, "9.30 s", "L3", "MATT"));
            Assert.AreEqual(3, Records.Submit(TestEvent, 9.3, true, "9.30 s", "L4", "MATT"), "a tie goes behind the older mark");
            Records.Submit(TestEvent, 9.9, true, "9.90 s", "L5", "MATT");
            Records.Submit(TestEvent, 9.8, true, "9.80 s", "L6", "MATT");
            Assert.AreEqual(0, Records.Submit(TestEvent, 10.5, true, "10.50 s", "L7", "MATT"), "outside the top 5");
            var top = Records.Top(TestEvent);
            CollectionAssert.AreEqual(new[] { 9.1, 9.3, 9.3, 9.5, 9.8 }, top.Select(e => e.value).ToArray());
            Assert.IsTrue(Records.TryGet(TestEvent, out var wr, out var text));
            Assert.AreEqual(9.1, wr);
            StringAssert.Contains("Z2", text);
            Assert.AreEqual(0.4, Records.GapToRecord(TestEvent, 9.5), 1e-9);
        }

        [Test]
        public void RecordsHigherIsBetterAndLegacyMigration()
        {
            Records.Clear(TestEvent);
            PlayerPrefs.SetString($"poolympic.record.{TestEvent:00}", "4.06|4.06 m/s · L7");   // pre-list format
            Assert.IsTrue(Records.TryGet(TestEvent, out var v, out var text));
            Assert.AreEqual(4.06, v, 1e-9);
            StringAssert.Contains("L7", text);
            Assert.AreEqual(2, Records.Submit(TestEvent, 4.01, false, "4.01 m/s", "L1"), "slower: #2, not a record");
            Assert.AreEqual(1, Records.Submit(TestEvent, 4.2, false, "4.20 m/s", "L2"));
        }

        [Test]
        public void ConvexHullAndSignedDistance()
        {
            var pts = new List<Vector2> { new(0, 0), new(1, 0), new(1, 1), new(0, 1), new(0.5f, 0.5f), new(0.2f, 0.9f) };
            var hull = Geometry2D.ConvexHull(pts);
            Assert.AreEqual(4, hull.Count, "interior points are dropped");
            Assert.AreEqual(0.25f, Geometry2D.SignedDistance(hull, new Vector2(0.25f, 0.5f)), 1e-5f, "inside: distance to the nearest edge");
            Assert.AreEqual(-0.5f, Geometry2D.SignedDistance(hull, new Vector2(1.5f, 0.5f)), 1e-5f, "outside: negative");
            var foot = Geometry2D.ConvexHull(new List<Vector2> { new(0, 0), new(0.2f, 0) });
            Assert.AreEqual(2, foot.Count);
            Assert.Less(Geometry2D.SignedDistance(foot, new Vector2(0.1f, 0f)), 1e-5f, "a line support has no inside");
        }

        [Test]
        public void EveryEventBrainHasACalibratedCritic()
        {
            var json = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/PoOlympic/Models/confidence_model.json");
            Assert.IsNotNull(json, "run training/tools/fit_confidence.py");
            foreach (var brain in new[] { "r0_v2_it1000", "rung2", "r2f_v3_it100", "zombie_rung0", "zombie_rung2", "crawl_matt", "crawl_zombie",
                                         "grandma_rung0", "grandma_rung2", "crawl_grandma" })
            {
                var e = BrainConfidence.Find(json, brain);
                Assert.IsNotNull(e, $"{brain}: no calibrated confidence entry");
                Assert.Greater(e.auc, 0.6f, brain);
                Assert.IsNotNull(AssetDatabase.LoadAssetAtPath<Object>($"Assets/PoOlympic/Models/Brains/{brain}.critic.onnx"), $"{brain}.critic.onnx missing (tools/export_critic.py + sync)");
                // a sudden value drop means less confidence
                Assert.Less(e.Probability(5f, -3f, 1f), e.Probability(5f, 0f, 1f) + 1e-6f, brain);
            }
        }

        [Test]
        public void EventScenesCarryTheBroadcastFxLayer()
        {
            var current = EditorSceneManager.GetActiveScene().path;
            try
            {
                foreach (var path in Editor.EventScenes.EventScenePaths.Values.Distinct())
                {
                    EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                    var director = Object.FindAnyObjectByType<BroadcastDirector>(FindObjectsInactive.Include);
                    Assert.IsNotNull(director, path);
                    Assert.IsTrue(director.wide && director.headOn && director.high && director.hot && director.winner, $"{path}: Cinemachine rig");
                    Assert.IsNotNull(director.tension, $"{path}: TensionMeter");
                    Assert.IsNotNull(Object.FindAnyObjectByType<ImpactFx>(FindObjectsInactive.Include), $"{path}: ImpactFx");
                    var audio = Object.FindAnyObjectByType<ArenaAudio>(FindObjectsInactive.Include);
                    Assert.IsTrue(audio != null && audio.bed != null && audio.bed.clip != null && audio.voices.Length > 0, $"{path}: ArenaAudio");
                    foreach (var r in Object.FindObjectsByType<PolicyRunner>(FindObjectsInactive.Include))
                    {
                        Assert.IsNotNull(r.GetComponent<AthleteTelemetry>(), $"{path}: {r.name} telemetry");
                        Assert.IsNotNull(r.critic, $"{path}: {r.name} critic");
                    }
                    // athlete fill lights: on the athletes' rendering layer only, and every athlete mesh carries it
                    var fill = GameObject.Find("AthleteFill");
                    Assert.IsNotNull(fill, $"{path}: AthleteFill rig");
                    foreach (var l in fill.GetComponentsInChildren<Light>())
                        Assert.AreEqual((int)Editor.StadiumLook.AthleteLayer, l.renderingLayerMask, $"{path}: {l.name} must not light the stadium");
                    foreach (var r in Object.FindObjectsByType<SkinnedMeshRenderer>(FindObjectsInactive.Include))
                        Assert.AreNotEqual(0u, r.renderingLayerMask & Editor.StadiumLook.AthleteLayer, $"{path}: {r.name} not on the athlete light layer");
                    Assert.IsEmpty(Object.FindObjectsByType<Collider>(FindObjectsInactive.Include), $"{path}: PhysX collider");
                    Assert.IsEmpty(Object.FindObjectsByType<Rigidbody>(FindObjectsInactive.Include), $"{path}: PhysX rigidbody");
                }
            }
            finally
            {
                if (!string.IsNullOrEmpty(current) && File.Exists(current)) EditorSceneManager.OpenScene(current, OpenSceneMode.Single);
            }
        }
    }
}
