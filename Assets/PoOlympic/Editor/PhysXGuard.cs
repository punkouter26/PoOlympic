using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Physics-engine isolation mandate (DESIGN.md §1): athlete scenes must contain no PhysX components.
    /// All physical interaction goes through MuJoCo (MjBody/MjGeom/MjFreeJoint).
    /// Enforced on menu, before entering Play mode and before every Player build.
    /// </summary>
    [InitializeOnLoad]
    public static class PhysXGuard
    {
        public const string AthleteScenesFolder = "Assets/PoOlympic/Scenes";

        static readonly Type[] Forbidden =
        {
            typeof(Rigidbody), typeof(Collider), typeof(CharacterController), typeof(Joint),
            typeof(ArticulationBody), typeof(ConstantForce), typeof(Cloth), typeof(WheelCollider),
            typeof(Rigidbody2D), typeof(Collider2D), typeof(Joint2D),
        };

        static PhysXGuard()
        {
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        public static bool IsAthleteScene(Scene scene) =>
            scene.path.StartsWith(AthleteScenesFolder + "/", StringComparison.Ordinal);

        /// <summary>Returns "path/to/object: ComponentType" for every forbidden component in the scene.</summary>
        public static List<string> FindViolations(Scene scene)
        {
            var hits = new List<string>();
            foreach (var root in scene.GetRootGameObjects())
            foreach (var t in Forbidden)
            foreach (var c in root.GetComponentsInChildren(t, true))
                hits.Add($"{HierarchyPath(c.transform)}: {c.GetType().Name}");
            return hits;
        }

        static string HierarchyPath(Transform t) =>
            t.parent == null ? t.name : HierarchyPath(t.parent) + "/" + t.name;

        [MenuItem("PoOlympic/Validate PhysX Isolation")]
        public static bool ValidateOpenScenes()
        {
            var all = new List<string>();
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                var s = SceneManager.GetSceneAt(i);
                if (IsAthleteScene(s)) all.AddRange(FindViolations(s).Select(v => $"{s.name} › {v}"));
            }
            if (all.Count == 0)
            {
                Debug.Log("[PhysXGuard] OK — no PhysX components in open athlete scenes.");
                return true;
            }
            Debug.LogError("[PhysXGuard] PhysX components are forbidden in athlete scenes:\n" + string.Join("\n", all));
            return false;
        }

        static void OnPlayModeStateChanged(PlayModeStateChange change)
        {
            if (change != PlayModeStateChange.ExitingEditMode) return;
            EnforceProjectPhysicsSettings();
            if (!ValidateOpenScenes()) EditorApplication.isPlaying = false;
        }

        /// <summary>PhysX must never step: its simulation mode is Script and nothing calls Physics.Simulate.</summary>
        public static void EnforceProjectPhysicsSettings()
        {
            if (Physics.simulationMode != SimulationMode.Script)
            {
                Physics.simulationMode = SimulationMode.Script;
                Debug.Log("[PhysXGuard] Physics.simulationMode forced to Script.");
            }
            Physics.autoSyncTransforms = false;
        }

        class BuildCheck : IPreprocessBuildWithReport
        {
            public int callbackOrder => 0;

            public void OnPreprocessBuild(BuildReport report)
            {
                EnforceProjectPhysicsSettings();
                var bad = new List<string>();
                foreach (var scene in EditorBuildSettings.scenes.Where(s => s.enabled && s.path.StartsWith(AthleteScenesFolder)))
                {
                    var s = EditorSceneManager.OpenScene(scene.path, OpenSceneMode.Additive);
                    bad.AddRange(FindViolations(s).Select(v => $"{s.name} › {v}"));
                    EditorSceneManager.CloseScene(s, true);
                }
                if (bad.Count > 0)
                    throw new BuildFailedException("PhysX components are forbidden in athlete scenes:\n" + string.Join("\n", bad));
            }
        }
    }
}
