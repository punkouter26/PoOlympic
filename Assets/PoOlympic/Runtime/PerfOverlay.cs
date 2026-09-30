using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.LowLevel;
using UnityEngine.PlayerLoop;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// GFX idea 10 — real-time performance telemetry + automatic quality steps.
    ///   overlay   one line: fps · CPU / GPU frame ms (FrameTimingManager) · sim ms (physics + brains: every
    ///             MonoBehaviour FixedUpdate of the frame, bracketed in the player loop) · draw calls (editor / dev
    ///             builds) · Android thermal status + battery °C · quality tier. Toggle: F3, a three-finger tap, or tap
    ///             the version label of the broadcast HUD (BroadcastHud.ToggleOverlay).
    ///   log       one CSV row per second to persistentDataPath/perf/&lt;scene&gt;_&lt;time&gt;.csv (the Android re-measure).
    ///   auto      on a phone (autoQuality): sustained throttling (thermal ≥ moderate, or fps under 90 % of the target
    ///             for 5 s) steps the tier up, 20 s of cool, full-rate running steps it back down:
    ///               1  render scale −0.1
    ///               2  + light beams off (_PoBeamDim), screen feed camera at a third of the rate
    ///               3  + brain critics at 5 Hz (PolicyRunner.criticEvery 10), shadow distance halved
    ///             Nothing here touches ctrl or the physics (parity untouched).
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    [DefaultExecutionOrder(200)]
    public class PerfOverlay : MonoBehaviour
    {
        public StyleSheet style;
        public bool autoQuality = true;
        [Tooltip("Log a CSV row per second (also while the overlay is hidden).")]
        public bool logCsv = true;
        public ScreenFeed screens;

        public static PerfOverlay Instance { get; private set; }
        public int Tier { get; private set; }
        public bool Visible { get => _visible; set { _visible = value; if (_label != null) _label.style.display = value ? DisplayStyle.Flex : DisplayStyle.None; PlayerPrefs.SetInt(PrefKey, value ? 1 : 0); } }

        const string PrefKey = "poolympic.perfOverlay";
        static readonly Stopwatch SimWatch = new();
        static double _simMsAccum;
        static bool _loopPatched;

        Label _label;
        bool _visible;
        readonly FrameTiming[] _timing = new FrameTiming[1];
        ProfilerRecorder _draws;
        float _acc, _frames, _cpu, _gpu, _fpsAvg = 60f, _hotFor, _coolFor, _lastStep = -99f;
#if UNITY_ANDROID && !UNITY_EDITOR
        float _thermalAt = -99f;
#endif
        int _thermal = -1;
        float _batteryC = float.NaN;
        StreamWriter _csv;
        float _baseScale = -1f, _baseShadow = -1f;

        void OnEnable()
        {
            Instance = this;
            PatchPlayerLoop();
            var root = GetComponent<UIDocument>().rootVisualElement;
            root.Clear();
            if (style != null) root.styleSheets.Add(style);
            root.pickingMode = PickingMode.Ignore;
            _label = new Label("") { pickingMode = PickingMode.Ignore };
            _label.AddToClassList("perf-line");
            root.Add(_label);
            _visible = PlayerPrefs.GetInt(PrefKey, 0) == 1;
            _label.style.display = _visible ? DisplayStyle.Flex : DisplayStyle.None;
            _draws = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
            if (logCsv) OpenCsv();
        }

        void OnDisable()
        {
            if (Instance == this) Instance = null;
            _draws.Dispose();
            _csv?.Dispose();
            _csv = null;
            if (Tier != 0) SetTier(0);        // leave the render pipeline asset as it was
        }

        public static void Toggle() { if (Instance != null) Instance.Visible = !Instance.Visible; }

        // -------------------------------------------------------------------------------------- sim timing
        struct SimBegin { }
        struct SimEnd { }

        /// <summary>Wrap the FixedUpdate script pass (MuJoCo step + PolicyRunner brains) with a stopwatch.</summary>
        static void PatchPlayerLoop()
        {
            if (_loopPatched) return;
            var loop = PlayerLoop.GetCurrentPlayerLoop();
            for (int i = 0; i < loop.subSystemList.Length; i++)
            {
                if (loop.subSystemList[i].type != typeof(FixedUpdate)) continue;
                var fixedLoop = loop.subSystemList[i];
                var list = fixedLoop.subSystemList.ToList();
                int k = list.FindIndex(s => s.type == typeof(FixedUpdate.ScriptRunBehaviourFixedUpdate));
                if (k < 0) return;
                list.Insert(k + 1, new PlayerLoopSystem { type = typeof(SimEnd), updateDelegate = () => { SimWatch.Stop(); _simMsAccum += SimWatch.Elapsed.TotalMilliseconds; } });
                list.Insert(k, new PlayerLoopSystem { type = typeof(SimBegin), updateDelegate = () => SimWatch.Restart() });
                fixedLoop.subSystemList = list.ToArray();
                loop.subSystemList[i] = fixedLoop;
                PlayerLoop.SetPlayerLoop(loop);
                _loopPatched = true;
                return;
            }
        }

        // -------------------------------------------------------------------------------------- per frame
        void Update()
        {
            if (Keyboard.current != null && Keyboard.current.f3Key.wasPressedThisFrame) Toggle();
            var ts = Touchscreen.current;
            if (ts != null && ts.touches.Count(t => t.press.isPressed) >= 3 && ts.touches.Any(t => t.press.wasPressedThisFrame)) Toggle();

            FrameTimingManager.CaptureFrameTimings();
            if (FrameTimingManager.GetLatestTimings(1, _timing) > 0)
            {
                _cpu += (float)_timing[0].cpuFrameTime;
                _gpu += (float)_timing[0].gpuFrameTime;
            }
            _acc += Time.unscaledDeltaTime;
            _frames++;
            if (_acc < 1f) return;

            float fps = _frames / _acc, cpu = _cpu / _frames, gpu = _gpu / _frames, sim = (float)(_simMsAccum / _frames);
            _acc = _frames = _cpu = _gpu = 0f;
            _simMsAccum = 0;
            _fpsAvg = Mathf.Lerp(_fpsAvg, fps, 0.5f);
            ReadThermal();
            long draws = _draws.Valid ? _draws.LastValue : -1;
            if (autoQuality) AutoQuality(fps);
            if (_visible)
                _label.text = $"{fps:0} fps · CPU {cpu:0.0} · GPU {(gpu > 0 ? gpu.ToString("0.0") : "–")} · sim {sim:0.00} ms · " +
                              $"{(draws > 0 ? draws + " draws" : "draws n/a")} · {ThermalText()}{(float.IsNaN(_batteryC) ? "" : $" {_batteryC:0}°C")} · Q{Tier}";
            _csv?.WriteLine(FormattableString.Invariant($"{Time.realtimeSinceStartup:0.0},{fps:0.0},{cpu:0.00},{gpu:0.00},{sim:0.000},{draws},{_thermal},{_batteryC:0.0},{Tier}"));
        }

        string ThermalText() => _thermal switch
        {
            < 0 => "thermal n/a",
            0 => "cool",
            1 => "light",
            2 => "moderate",
            3 => "severe",
            _ => "critical",
        };

        void AutoQuality(float fps)
        {
            if (!Application.isMobilePlatform) return;
            float target = Application.targetFrameRate > 0 ? Application.targetFrameRate : 60f;
            bool hot = _thermal >= 2 || fps < target * 0.9f;
            bool cool = _thermal <= 1 && fps >= target * 0.97f;
            _hotFor = hot ? _hotFor + 1f : 0f;
            _coolFor = cool ? _coolFor + 1f : 0f;
            float now = Time.unscaledTime;
            if (_hotFor >= 5f && Tier < 3 && now - _lastStep > 10f) { SetTier(Tier + 1); _lastStep = now; _hotFor = 0f; }
            else if (_coolFor >= 20f && Tier > 0 && now - _lastStep > 10f) { SetTier(Tier - 1); _lastStep = now; _coolFor = 0f; }
        }

        void SetTier(int tier)
        {
            Tier = Mathf.Clamp(tier, 0, 3);
            if (GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset urp)
            {
                if (_baseScale < 0f) { _baseScale = urp.renderScale; _baseShadow = urp.shadowDistance; }
                urp.renderScale = Mathf.Max(0.5f, _baseScale - (Tier >= 1 ? 0.1f : 0f));
                urp.shadowDistance = Tier >= 3 ? _baseShadow * 0.5f : _baseShadow;
            }
            Shader.SetGlobalFloat("_PoBeamDim", Tier >= 2 ? 1f : 0f);
            if (screens != null) screens.camEvery = Tier >= 2 ? 6 : 3;
            foreach (var r in FindObjectsByType<PolicyRunner>()) r.criticEvery = Tier >= 3 ? 10 : 5;
        }

        // -------------------------------------------------------------------------------------- device
        void ReadThermal()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (Time.unscaledTime - _thermalAt < 2f) return;
            _thermalAt = Time.unscaledTime;
            try
            {
                using var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
                using var activity = player.GetStatic<AndroidJavaObject>("currentActivity");
                using var power = activity.Call<AndroidJavaObject>("getSystemService", "power");
                _thermal = power.Call<int>("getCurrentThermalStatus");            // API 29+
                using var filter = new AndroidJavaObject("android.content.IntentFilter", "android.intent.action.BATTERY_CHANGED");
                using var intent = activity.Call<AndroidJavaObject>("registerReceiver", null, filter);
                if (intent != null) _batteryC = intent.Call<int>("getIntExtra", "temperature", -1000) / 10f;
            }
            catch (Exception) { _thermal = -1; }
#endif
        }

        void OpenCsv()
        {
            try
            {
                var dir = Path.Combine(Application.persistentDataPath, "perf");
                Directory.CreateDirectory(dir);
                var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
                _csv = new StreamWriter(Path.Combine(dir, $"{scene}_{DateTime.Now:yyyyMMdd_HHmmss}.csv")) { AutoFlush = true };
                _csv.WriteLine("t_s,fps,cpu_ms,gpu_ms,sim_ms,draw_calls,thermal,battery_c,tier");
            }
            catch (Exception e) { UnityEngine.Debug.LogWarning("[PerfOverlay] no CSV: " + e.Message); _csv = null; }
        }
    }
}
