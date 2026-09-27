using NUnit.Framework;
using PoOlympic.Editor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Tests
{
    public class PhysXGuardTests
    {
        [Test]
        public void CleanSceneHasNoViolations()
        {
            // the test runner restores the user's scenes afterwards
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            new GameObject("athlete").AddComponent<Mujoco.MjBody>();
            Assert.IsEmpty(PhysXGuard.FindViolations(scene));
        }

        [Test]
        public void InjectedBoxColliderIsDetected()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var parent = new GameObject("athlete");
            var child = GameObject.CreatePrimitive(PrimitiveType.Cube); // CreatePrimitive adds a BoxCollider
            child.transform.SetParent(parent.transform);
            child.AddComponent<Rigidbody>();
            var hits = PhysXGuard.FindViolations(scene);
            Assert.That(hits, Has.Some.Contains("BoxCollider"));
            Assert.That(hits, Has.Some.Contains("Rigidbody"));
        }

        [Test]
        public void ProjectPhysicsNeverSimulates()
        {
            PhysXGuard.EnforceProjectPhysicsSettings();
            Assert.AreEqual(SimulationMode.Script, Physics.simulationMode);
        }
    }
}
