using System;
using System.IO;
using System.Security.Cryptography;
using Mujoco;
using UnityEditor;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>
    /// B3 — Brings the generated MJCF (single source of truth: training/assets/scene_matt.xml) into Unity
    /// through the MuJoCo plug-in's own importer. The file is copied verbatim and its SHA-256 recorded, so the
    /// scene always knows which physics model it was built from.
    /// </summary>
    public static class AthleteImport
    {
        public const string SourceRelative = "training/assets/scene_matt.xml";
        public const string ModelsFolder = "Assets/PoOlympic/Models";
        public const string SceneRootName = "MuJoCoScene";

        static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

        public static string Sha256(string path)
        {
            using var sha = SHA256.Create();
            using var fs = File.OpenRead(path);
            return BitConverter.ToString(sha.ComputeHash(fs)).Replace("-", "").ToLowerInvariant();
        }

        /// <summary>Copies scene_matt.xml into Assets and returns (asset path, sha256).</summary>
        public static (string path, string sha) SyncModel()
        {
            var src = Path.Combine(ProjectRoot, SourceRelative);
            if (!File.Exists(src)) throw new FileNotFoundException("Generated MJCF not found — run training/tools/build_mjcf.py", src);
            Directory.CreateDirectory(Path.Combine(ProjectRoot, ModelsFolder));
            var dst = Path.Combine(ProjectRoot, ModelsFolder, "scene_matt.xml");
            File.Copy(src, dst, overwrite: true);
            var sha = Sha256(dst);
            if (sha != Sha256(src)) throw new IOException("MJCF copy hash mismatch");
            AssetDatabase.ImportAsset($"{ModelsFolder}/scene_matt.xml");
            return (dst, sha);
        }

        [MenuItem("PoOlympic/Import Athlete Scene (MJCF)")]
        public static GameObject ImportIntoActiveScene()
        {
            var (path, sha) = SyncModel();
            var existing = GameObject.Find(SceneRootName);
            if (existing != null) UnityEngine.Object.DestroyImmediate(existing);

            // ImportString on the original text, NOT ImportFile: ImportFile round-trips the model through
            // mj_saveLastXML, which prints 6 significant digits and converts angles to radians (G0 drift ~2e-6).
            // Our generated MJCF is already flat, in degrees, with float32-safe 7-digit numbers.
            var root = new MjImporterWithAssets().ImportString(File.ReadAllText(path), "scene_matt", path);
            if (root == null) throw new Exception("MuJoCo importer failed — see console");
            root.name = SceneRootName;
            var prov = root.AddComponent<ModelProvenance>();
            prov.sourceFile = SourceRelative;
            prov.sha256 = sha;
            prov.importedUtc = DateTime.UtcNow.ToString("o");
            Undo.RegisterCreatedObjectUndo(root, "Import athlete MJCF");
            Debug.Log($"[AthleteImport] imported {SourceRelative} sha256={sha}");
            return root;
        }
    }
}
