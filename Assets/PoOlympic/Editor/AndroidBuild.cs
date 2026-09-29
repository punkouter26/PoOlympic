using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Android (arm64) player build. MuJoCo for Android = libmujoco.so built from the SAME MuJoCo release as the
    /// Windows mujoco.dll / org.mujoco package (3.11.0; the C# bindings mirror its struct layouts), with the NDK that
    /// ships with Unity: see docs / tasks.md "Android". The prebuilt joanllobera/mujoco-bin library is 3.5.0 and must not
    /// be mixed with the 3.11 plugin.
    /// </summary>
    public static class AndroidBuild
    {
        public const string AndroidLib = "Assets/Plugins/MuJoCo/Android/arm64-v8a/libmujoco.so";
        public const string WindowsDll = "Assets/Plugins/MuJoCo/mujoco.dll";
        public const string ApkPath = "Builds/Android/PoOlympic.apk";
        public const string PackageId = "com.poolympic.game";

        /// <summary>Native plugin import settings: the .so only on Android ARM64, the .dll only in the Editor / Windows.</summary>
        [MenuItem("PoOlympic/Android/Configure MuJoCo plugins")]
        public static string ConfigurePlugins()
        {
            var so = (PluginImporter)AssetImporter.GetAtPath(AndroidLib) ?? throw new FileNotFoundException(AndroidLib);
            so.SetCompatibleWithAnyPlatform(false);
            so.SetCompatibleWithEditor(false);
            so.SetCompatibleWithPlatform(BuildTarget.Android, true);
            so.SetPlatformData(BuildTarget.Android, "CPU", "ARM64");
            so.SaveAndReimport();

            var dll = (PluginImporter)AssetImporter.GetAtPath(WindowsDll) ?? throw new FileNotFoundException(WindowsDll);
            dll.SetCompatibleWithAnyPlatform(false);
            dll.SetCompatibleWithEditor(true);
            dll.SetEditorData("OS", "Windows");
            dll.SetEditorData("CPU", "x86_64");
            dll.SetCompatibleWithPlatform(BuildTarget.StandaloneWindows64, true);
            dll.SetCompatibleWithPlatform(BuildTarget.Android, false);
            dll.SaveAndReimport();
            return "plugins configured";
        }

        public const string ParityScene = "Assets/PoOlympic/Scenes/Parity_Device.unity";
        public const string ParityApkPath = "Builds/Android/PoOlympic_parity.apk";
        public const string ParityPackageId = "com.poolympic.parity";
        public const string ParityRecordName = "pedestal_rung0_android";

        /// <summary>
        /// ARM↔desktop parity + mobile perf APK (own package id, the game stays installed): Parity_Device.unity = the solo
        /// pedestal scene with the event controller off, the PolicyRunner on the standard parity script recording 250
        /// ticks (= training/parity reference "pedestal_rung0"), plus DeviceProbe, which then loads the Steeplechase heat
        /// for the perf sample. Results: /sdcard/Android/data/com.poolympic.parity/files/ (adb pull).
        /// </summary>
        [MenuItem("PoOlympic/Android/Build parity + perf APK")]
        public static string BuildParityApk()
        {
            if (EditorApplication.isPlaying) throw new InvalidOperationException("exit Play mode first");
            var scene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene(EventScenes.IronPedestalScene);
            var runner = UnityEngine.Object.FindFirstObjectByType<PolicyRunner>();
            runner.useStandardParityScript = true;
            runner.recordName = ParityRecordName;
            runner.recordTicks = 250;
            UnityEngine.Object.FindFirstObjectByType<IronPedestalEvent>().gameObject.SetActive(false);
            var hud = GameObject.Find("EventHUD");
            if (hud != null) hud.SetActive(false);
            new GameObject("DeviceProbe").AddComponent<DeviceProbe>();
            UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene, ParityScene);

            string id = PlayerSettings.GetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Android), product = PlayerSettings.productName;
            try
            {
                PlayerSettings.SetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Android, ParityPackageId);
                PlayerSettings.productName = "PoOlympics Parity";
                return BuildScenes(new[] { ParityScene, EventScenes.SteepleScene }, ParityApkPath);
            }
            finally
            {
                PlayerSettings.SetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Android, id);
                PlayerSettings.productName = product;
            }
        }

        [MenuItem("PoOlympic/Android/Build APK (arm64)")]
        public static string Build()
        {
            PlayerSettings.SetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Android, PackageId);
            PlayerSettings.productName = "PoOlympics";
            var scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
            if (scenes.Length == 0) throw new InvalidOperationException("no scenes in Build Settings (EventScenes.SetBuildScenes)");
            return BuildScenes(scenes, ApkPath);
        }

        static string BuildScenes(string[] scenes, string apkPath)
        {
            ConfigurePlugins();
            PlayerSettings.SetScriptingBackend(UnityEditor.Build.NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel28;   // libmujoco.so needs aligned_alloc (API 28)
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;   // 9:16 broadcast framing

            Directory.CreateDirectory(Path.GetDirectoryName(apkPath));
            File.WriteAllText("Builds/Android/last_build.txt", $"building {apkPath} ...");
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = apkPath,
                target = BuildTarget.Android,
                options = BuildOptions.None,
            });
            var s = report.summary;
            string msg = $"{s.result}: {apkPath} {s.totalSize / 1e6:F1} MB, {s.totalErrors} errors, {s.totalTime.TotalMinutes:F1} min, {scenes.Length} scenes";
            File.WriteAllText("Builds/Android/last_build.txt", msg + Environment.NewLine +
                string.Join(Environment.NewLine, report.steps.SelectMany(st => st.messages).Where(m => m.type == LogType.Error || m.type == LogType.Exception).Select(m => m.content)));
            Debug.Log("[AndroidBuild] " + msg);
            return msg;
        }
    }
}
