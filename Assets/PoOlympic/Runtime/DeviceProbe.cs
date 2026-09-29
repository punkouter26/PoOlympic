using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Mujoco;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic
{
    /// <summary>
    /// Android parity + perf probe (tasks.md backlog "Android"; lives only in the parity APK, AndroidBuild.BuildParityApk):
    ///   1. ARM↔desktop parity: the pedestal G5 attempt (standard shove + cube script, Rung 0 brain) recorded by the
    ///      scene's PolicyRunner → persistentDataPath/unity_run_&lt;recordName&gt;.json (pulled with adb, compared on the
    ///      desktop with tools/compare_closed_loop.py against the CPU reference)
    ///   2. perf: mj_step timing on a copy of the live data + frame-time percentiles while a real 8-athlete heat
    ///      (perfScene, betting window off) runs → persistentDataPath/device_probe.json; "[DeviceProbe] DONE" in logcat
    /// </summary>
    public class DeviceProbe : MonoBehaviour
    {
        public string perfScene = "Event_SteeplechaseJog";
        public float perfSeconds = 20f, warmupSeconds = 6f;
        readonly StringBuilder _json = new("{");

        IEnumerator Start()
        {
            DontDestroyOnLoad(gameObject);
            Application.targetFrameRate = 60;
            Add("device", SystemInfo.deviceModel);
            Add("cpu", $"{SystemInfo.processorType} x{SystemInfo.processorCount}");
            Add("gpu", SystemInfo.graphicsDeviceName);
            Add("os", SystemInfo.operatingSystem);

            // 1. parity recording (the scene's runner records recordTicks control ticks, then flushes)
            var runner = FindFirstObjectByType<PolicyRunner>();
            float t0 = Time.realtimeSinceStartup;
            while (runner != null && !runner.RecordingDone && Time.realtimeSinceStartup - t0 < 60f) yield return null;
            Add("parity_record", runner != null && runner.RecordingDone ? runner.RecordPath : "timeout");
            Add("parity_seconds_wall", Time.realtimeSinceStartup - t0);
            Add("mj_step_ms_solo", BenchStep(400));

            // 2. perf in a real heat
            SceneManager.sceneLoaded += OnLoaded;
            SceneManager.LoadScene(perfScene);
            yield return new WaitForSecondsRealtime(warmupSeconds);
            Add("mj_step_ms_meet", BenchStep(400));
            var dts = new List<float>();
            float end = Time.realtimeSinceStartup + perfSeconds;
            while (Time.realtimeSinceStartup < end)
            {
                dts.Add(Time.unscaledDeltaTime * 1000f);
                yield return null;
            }
            dts.Sort();
            float P(float q) => dts[Mathf.Clamp((int)(q * (dts.Count - 1)), 0, dts.Count - 1)];
            Add("frames", dts.Count);
            Add("fps_mean", 1000f / dts.Average());
            Add("frame_ms_p50", P(0.5f));
            Add("frame_ms_p95", P(0.95f));
            Add("frame_ms_p99", P(0.99f));
            Add("frame_ms_max", dts[^1]);
            var race = FindFirstObjectByType<TrackRaceEvent>();
            if (race != null) Add("heat_leader_x_m", race.LeaderX);
            _json.Append("\"done\":true}");
            var path = Path.Combine(Application.persistentDataPath, "device_probe.json");
            File.WriteAllText(path, _json.ToString());
            Debug.Log($"[DeviceProbe] DONE {path} {_json}");
        }

        void OnLoaded(Scene s, LoadSceneMode mode)
        {
            SceneManager.sceneLoaded -= OnLoaded;
            foreach (var hud in FindObjectsByType<BroadcastHud>(FindObjectsSortMode.None)) hud.bettingWindow = false;
        }

        static unsafe double BenchStep(int n)
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null) return -1;
            var m = MjScene.Instance.Model;
            var d = MujocoLib.mj_makeData(m);
            MujocoLib.mj_copyData(d, m, MjScene.Instance.Data);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < n; i++) MujocoLib.mj_step(m, d);
            sw.Stop();
            MujocoLib.mj_deleteData(d);
            return sw.Elapsed.TotalMilliseconds / n;
        }

        void Add(string key, object v)
        {
            string val = v switch
            {
                string s => $"\"{s.Replace("\\", "/").Replace("\"", "'")}\"",
                float f => f.ToString("R", CultureInfo.InvariantCulture),
                double x => x.ToString("R", CultureInfo.InvariantCulture),
                _ => System.Convert.ToString(v, CultureInfo.InvariantCulture),
            };
            _json.Append('"').Append(key).Append("\":").Append(val).Append(',');
        }
    }
}
