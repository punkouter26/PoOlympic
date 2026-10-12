using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic.Tests
{
    /// <summary>
    /// Every scene of the player build is whole: no missing scripts, no reference to an object that is gone, the UI
    /// bound to what it needs (HUD → board + style sheet, menu → its UXML and loadable event scenes), and no IMGUI
    /// (OnGUI) screen — those are dev screens and stay out of the build (EventScenes.SetBuildScenes).
    /// </summary>
    public class SceneIntegrityTests
    {
        public static IEnumerable<string> BuildScenes() => EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path);

        string _before;

        [OneTimeSetUp] public void Remember() => _before = EditorSceneManager.GetActiveScene().path;

        [OneTimeTearDown]
        public void Restore()
        {
            if (!string.IsNullOrEmpty(_before) && File.Exists(_before)) EditorSceneManager.OpenScene(_before, OpenSceneMode.Single);
        }

        static string PathOf(Transform t) => t.parent == null ? t.name : PathOf(t.parent) + "/" + t.name;

        [Test]
        public void MenuIsTheStartScene() =>
            Assert.AreEqual(MeetLineup.MenuScene, Path.GetFileNameWithoutExtension(BuildScenes().FirstOrDefault() ?? ""));

        [TestCaseSource(nameof(BuildScenes))]
        public void BuildSceneIsWhole(string path)
        {
            Assert.IsTrue(File.Exists(path), "in Build Settings but not on disk");
            var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
            var built = new HashSet<string>(BuildScenes().Select(Path.GetFileNameWithoutExtension));
            var bad = new List<string>();
            const BindingFlags any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

            foreach (var t in scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)))
            {
                int missing = GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(t.gameObject);
                if (missing > 0) bad.Add($"{missing} missing script(s): {PathOf(t)}");
                foreach (var c in t.GetComponents<Component>())
                {
                    if (c == null) continue;
                    var type = c.GetType();
                    string where = $"{PathOf(t)} ({type.Name})";
                    if (type.Namespace == "PoOlympic" && type.GetMethod("OnGUI", any) != null) bad.Add($"IMGUI screen in a build scene: {where}");

                    // a reference that points at an object which no longer exists (not one left empty on purpose)
                    using var so = new SerializedObject(c);
                    for (var p = so.GetIterator(); p.NextVisible(true);)
                        if (p.propertyType == SerializedPropertyType.ObjectReference && p.objectReferenceValue == null && p.objectReferenceEntityIdValue != EntityId.None)
                            bad.Add($"broken reference: {where}.{p.propertyPath}");

                    switch (c)
                    {
                        case UIDocument doc when doc.panelSettings == null: bad.Add($"UIDocument without panel settings: {where}"); break;
                        case BroadcastHud hud:
                            if (!(hud.board is IBroadcastBoard)) bad.Add($"HUD without an event board: {where}");
                            if (hud.style == null) bad.Add($"HUD without its style sheet: {where}");
                            break;
                        case PerfOverlay perf when perf.style == null: bad.Add($"PerfOverlay without its style sheet: {where}"); break;
                        case MainMenuController menu:
                            if (menu.GetComponent<UIDocument>().visualTreeAsset == null) bad.Add($"menu without its UXML: {where}");
                            if (menu.events.Count == 0) bad.Add($"menu without events: {where}");
                            foreach (var e in menu.events.Where(e => !built.Contains(e.scene)))
                                bad.Add($"menu event {e.number:00} loads '{e.scene}', which is not in Build Settings");
                            break;
                    }
                }
            }
            Assert.IsEmpty(bad, $"{path}:\n  " + string.Join("\n  ", bad.Take(30)));
        }
    }
}
