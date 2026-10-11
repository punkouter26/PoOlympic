using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Data logging (AGENTS.md): what every athlete did in every heat, for movement profiles and win / loss patterns.
    ///   &lt;event&gt;_h&lt;heat&gt;_&lt;time&gt;.csv.gz   one row per control tick per lane (50 Hz): tick, lane, command, pelvis
    ///                                     position, action_raw[n], ctrl[n] — written by PolicyRunner
    ///   heats.jsonl                       one line per finished heat: event, places, marks, traits and the telemetry
    ///                                     bests of each lane — written by BroadcastHud at the result
    /// Editor: parity/actions/unity (next to the Python gate logs, training/tools/report_actions.py reads both);
    /// player builds: persistentDataPath/actions. On when GameSettings.LogActions (default: editor only — a 30 s heat
    /// of 8 lanes is ~1 MB gzipped). The newest <see cref="KeepFiles"/> tick files are kept. Read-only on the
    /// simulation: nothing here feeds back into ctrl.
    /// </summary>
    public static class ActionLog
    {
        public const int KeepFiles = 40;
        static StreamWriter _w;
        static readonly StringBuilder Row = new(1024);
        static bool _header;

        public static bool Active => _w != null;

        public static string Dir => Application.isEditor
            ? Path.GetFullPath(Path.Combine(Application.dataPath, "..", "parity", "actions", "unity"))
            : Path.Combine(Application.persistentDataPath, "actions");

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => End();

        public static void BeginHeat(int eventNumber, int heat)
        {
            End();
            if (!GameSettings.LogActions || !Application.isPlaying) return;
            try
            {
                Directory.CreateDirectory(Dir);
                foreach (var old in new DirectoryInfo(Dir).GetFiles("*.csv.gz").OrderByDescending(f => f.LastWriteTimeUtc).Skip(KeepFiles - 1))
                    old.Delete();
                var path = Path.Combine(Dir, $"e{eventNumber:00}_h{heat}_{DateTime.Now:yyyyMMdd_HHmmss}.csv.gz");
                _w = new StreamWriter(new GZipStream(File.Create(path), System.IO.Compression.CompressionLevel.Fastest), Encoding.ASCII);
                _header = false;
            }
            catch (Exception e) { Debug.LogWarning("[ActionLog] off: " + e.Message); _w = null; }
        }

        public static void End()
        {
            try { _w?.Dispose(); } catch (IOException) { }
            _w = null;
        }

        /// <summary>One control tick of one lane (PolicyRunner, right after the brain ran).</summary>
        public static void Tick(int tick, string lane, Vector3 command, double x, double y, double z, float[] actionRaw, double[] ctrl)
        {
            if (_w == null) return;
            var inv = CultureInfo.InvariantCulture;
            if (!_header)
            {
                _header = true;
                Row.Clear().Append("tick,lane,cmd_vx,cmd_vy,cmd_wz,x,y,z");
                for (int i = 0; i < actionRaw.Length; i++) Row.Append(",a").Append(i);
                for (int i = 0; i < ctrl.Length; i++) Row.Append(",c").Append(i);
                _w.WriteLine(Row);
            }
            Row.Clear().Append(tick).Append(',').Append(lane)
               .Append(',').Append(command.x.ToString("0.###", inv)).Append(',').Append(command.y.ToString("0.###", inv))
               .Append(',').Append(command.z.ToString("0.###", inv))
               .Append(',').Append(x.ToString("0.###", inv)).Append(',').Append(y.ToString("0.###", inv)).Append(',').Append(z.ToString("0.###", inv));
            for (int i = 0; i < actionRaw.Length; i++) Row.Append(',').Append(actionRaw[i].ToString("0.####", inv));
            for (int i = 0; i < ctrl.Length; i++) Row.Append(',').Append(ctrl[i].ToString("0.####", inv));
            _w.WriteLine(Row);
        }

        [Serializable]
        class Lane
        {
            public string name, body, result, brain;
            public int place, latency;
            public bool bad;
            public float strength, noise, peakSpeed, peakPower, minConfidence, fatigue, overSpeedShare;
        }

        [Serializable]
        class Heat { public string time, scene; public int eventNumber, heat; public List<Lane> lanes = new(); }

        /// <summary>The heat's result line (BroadcastHud.Settle) and the end of its tick file.</summary>
        public static void EndHeat(int eventNumber, int heat, IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            bool on = GameSettings.LogActions && Application.isPlaying;
            End();
            if (!on) return;
            var h = new Heat
            {
                time = DateTime.Now.ToString("s", CultureInfo.InvariantCulture), eventNumber = eventNumber, heat = heat,
                scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
            };
            foreach (var r in rows)
            {
                var t = r.runner != null ? r.runner.GetComponent<AthleteTelemetry>() : null;
                h.lanes.Add(new Lane
                {
                    name = r.name, place = r.place, result = r.result, bad = r.bad, body = Odds.BodyOf(r.runner),
                    brain = r.runner != null && r.runner.brain != null ? r.runner.brain.name : "",
                    strength = r.runner != null ? r.runner.strength : 1f, latency = r.runner != null ? r.runner.latencySubsteps : 0,
                    noise = r.runner != null ? r.runner.obsNoise : 0f,
                    peakSpeed = t != null ? t.PeakSpeedMps : 0f, peakPower = t != null ? t.PeakPowerW : 0f,
                    minConfidence = t != null && !float.IsNaN(t.Confidence) ? t.MinConfidence : -1f,
                    fatigue = t != null ? t.Fatigue : 0f, overSpeedShare = t != null ? t.OverSpeedShare : 0f,
                });
            }
            try
            {
                Directory.CreateDirectory(Dir);
                File.AppendAllText(Path.Combine(Dir, "heats.jsonl"), JsonUtility.ToJson(h) + "\n");
            }
            catch (Exception e) { Debug.LogWarning("[ActionLog] heat line not written: " + e.Message); }
        }
    }
}
