using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Tests
{
    public class HandTunedSaveTests
    {
        const string Path = "Assets/PoOlympic/Tests/Editor/_handtuned_tmp.unity";

        [Test]
        public void RebuildKeepsHandPlacedObjects()
        {
            var current = EditorSceneManager.GetActiveScene().path;
            try
            {
                // the scene as a user left it: Rig/Cam moved by hand and marked, Rig/Other moved but not marked
                var a = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var rig = new GameObject("Rig");
                var cam = new GameObject("Cam"); cam.transform.SetParent(rig.transform);
                cam.transform.localPosition = new Vector3(1, 2, 3);
                cam.AddComponent<HandTuned>();
                var other = new GameObject("Other"); other.transform.SetParent(rig.transform);
                other.transform.localPosition = new Vector3(7, 7, 7);
                EditorSceneManager.SaveScene(a, Path);

                // the builder's fresh scene: everything at the origin, then saved over the old file
                var b = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var rig2 = new GameObject("Rig");
                new GameObject("Cam").transform.SetParent(rig2.transform);
                new GameObject("Other").transform.SetParent(rig2.transform);
                Assert.IsTrue(Editor.HandTunedSave.SaveBuilt(b, Path));

                var c = EditorSceneManager.OpenScene(Path, OpenSceneMode.Single);
                var root = c.GetRootGameObjects()[0].transform;
                Assert.AreEqual(new Vector3(1, 2, 3), root.Find("Cam").localPosition);
                Assert.IsNotNull(root.Find("Cam").GetComponent<HandTuned>());
                Assert.AreEqual(Vector3.zero, root.Find("Other").localPosition);   // unmarked: the build wins
            }
            finally
            {
                if (!string.IsNullOrEmpty(current)) EditorSceneManager.OpenScene(current, OpenSceneMode.Single);
                else EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                AssetDatabase.DeleteAsset(Path);
            }
        }
    }
}
