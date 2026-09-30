using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

namespace PoOlympic
{
    /// <summary>
    /// Demo (attract) mode, 2026-09-30: an endless Gauntlet with no human input. Each series = every playable event in a
    /// shuffled order with a fresh random lineup (DemoLineups); BroadcastHud advances each result after ResultSeconds;
    /// after the last event the series is scored into the season (DemoSeason) and the next series starts. Started by
    /// the main menu after IdleSeconds without input (or its menu sheet); any tap / key stops it and returns to the menu.
    /// DemoRunner (DontDestroyOnLoad) is the watchdog that keeps the loop moving when an event stalls.
    /// </summary>
    public static class DemoMode
    {
        public const float ResultSeconds = 6f;
        public static bool Active { get; private set; }
        /// <summary>This series' lineup theme ("Mixed", "All MATT", "4 v 4", …) for the HUD caption.</summary>
        public static string Theme { get; private set; } = "";
        public static int Series { get; private set; }

        static (int number, string scene)[] _events = Array.Empty<(int, string)>();
        static int _advancedAt = -1;                           // Gauntlet stage already advanced (one Advance per stage)

        public static void Begin(IEnumerable<(int number, string scene)> events)
        {
            _events = events.Where(e => Application.CanStreamedLevelBeLoaded(e.scene)).ToArray();
            if (_events.Length == 0) { Debug.LogWarning("[DemoMode] no loadable event scenes"); return; }
            Active = true;
            Series = 0;
            DemoRunner.Ensure();
            NextSeries();
        }

        public static void NextSeries()
        {
            Series++;
            var lineup = DemoLineups.Random(MeetLineup.Roster, out var theme);
            Theme = theme;
            var order = DemoLineups.Shuffle(_events);
            _advancedAt = -1;
            MeetLineup.Set(lineup, order[0].number);
            Debug.Log($"[DemoMode] series {Series}: {theme} ({string.Join(" ", lineup.Select(a => a[0]))}) · events {string.Join(" → ", order.Select(e => e.number))}");
            Gauntlet.Begin(order.Select(e => e.number).ToArray(), order.Select(e => e.scene).ToArray());
        }

        /// <summary>Move to the next stage (the HUD after the result, or the watchdog). Ignored if this stage already moved.</summary>
        public static void Advance(string why)
        {
            if (!Active || !Gauntlet.Active) return;
            int stage = Series * 1000 + Gauntlet.Index;
            if (_advancedAt == stage) return;
            _advancedAt = stage;
            if (why != null) Debug.LogWarning($"[DemoMode] event {MeetLineup.EventNumber} skipped: {why}");
            Gauntlet.Next();
        }

        /// <summary>Gauntlet.Next at the last stage: score the series into the season, start the next one.</summary>
        public static void SeriesFinished()
        {
            DemoSeason.Record(MeetLineup.Athletes, Enumerable.Range(0, 8).Select(Gauntlet.PointsOf).ToArray(), Gauntlet.Events.Length);
            Gauntlet.Abandon();
            NextSeries();
        }

        public static void Stop()
        {
            if (!Active) return;
            Active = false;
            Gauntlet.Abandon();
            Screen.sleepTimeout = SleepTimeout.SystemSetting;
            MeetLineup.ReturnToMenu();
        }
    }

    /// <summary>Random lineups for demo mode, weighted so most series mix bodies (the best contact moments).</summary>
    public static class DemoLineups
    {
        static readonly System.Random Rng = new();

        public static T[] Shuffle<T>(IEnumerable<T> items) => items.OrderBy(_ => Rng.Next()).ToArray();

        /// <summary>50 % mixed (every lane random, at least one of two bodies), 25 % all one body, 25 % rivalry
        /// (4 v 4 halves, or one odd athlete out among 7).</summary>
        public static string[] Random(IReadOnlyList<string> roster, out string theme)
        {
            var lanes = new string[8];
            double roll = Rng.NextDouble();
            if (roster.Count < 2 || (roll >= 0.5 && roll < 0.75))
            {
                string a = roster[Rng.Next(roster.Count)];
                for (int k = 0; k < 8; k++) lanes[k] = a;
                theme = $"All {a}";
                return lanes;
            }
            if (roll < 0.5)
            {
                do { for (int k = 0; k < 8; k++) lanes[k] = roster[Rng.Next(roster.Count)]; }
                while (lanes.Distinct().Count() < 2);
                theme = "Mixed lineup";
                return lanes;
            }
            var pair = Shuffle(roster).Take(2).ToArray();
            if (Rng.NextDouble() < 0.5)
            {
                for (int k = 0; k < 8; k++) lanes[k] = k < 4 ? pair[0] : pair[1];
                theme = $"4 v 4: {pair[0]} vs {pair[1]}";
            }
            else
            {
                int odd = Rng.Next(8);
                for (int k = 0; k < 8; k++) lanes[k] = k == odd ? pair[1] : pair[0];
                theme = $"One {pair[1]} among 7 {pair[0]}";
            }
            return lanes;
        }
    }

    /// <summary>
    /// Season table across demo series (PlayerPrefs, survives restarts): per athlete body the series won (body of the
    /// top lane), gauntlet points and lane-events run → points per event, fair across lineups with different counts.
    /// </summary>
    public static class DemoSeason
    {
        const string Key = "PoOlympic.DemoSeason";

        [Serializable] class Row { public string body; public int wins, points, laneEvents; }
        [Serializable] class Table { public int series; public List<Row> rows = new(); }

        static Table Load()
        {
            try { return JsonUtility.FromJson<Table>(PlayerPrefs.GetString(Key, "")) ?? new Table(); }
            catch (ArgumentException) { return new Table(); }
        }

        public static int SeriesPlayed => Load().series;

        public static void Record(string[] lineup, int[] lanePoints, int events)
        {
            var t = Load();
            t.series++;
            Row RowOf(string body)
            {
                var r = t.rows.Find(x => x.body == body);
                if (r == null) t.rows.Add(r = new Row { body = body });
                return r;
            }
            int best = Array.IndexOf(lanePoints, lanePoints.Max());
            RowOf(lineup[best]).wins++;
            for (int k = 0; k < lineup.Length; k++)
            {
                var r = RowOf(lineup[k]);
                r.points += lanePoints[k];
                r.laneEvents += events;
            }
            PlayerPrefs.SetString(Key, JsonUtility.ToJson(t));
            PlayerPrefs.Save();
        }

        /// <summary>"Season · 4 series · wins MATT 3 · ZOMBIE 1 · pts/event MATT 4.9 · ZOMBIE 3.8" ("" before the first).</summary>
        public static string Summary()
        {
            var t = Load();
            if (t.series == 0) return "";
            var rows = t.rows.OrderByDescending(r => r.wins).ToList();
            return $"Season · {t.series} series · wins " + string.Join(" · ", rows.Select(r => $"{r.body} {r.wins}")) +
                   " · pts/event " + string.Join(" · ", rows.Select(r => $"{r.body} {(r.laneEvents > 0 ? (float)r.points / r.laneEvents : 0f):0.0}"));
        }

        public static void Reset() { PlayerPrefs.DeleteKey(Key); PlayerPrefs.Save(); }
    }

    /// <summary>
    /// Demo mode's per-scene companion (DontDestroyOnLoad): stops the demo on any tap / key (after a short grace
    /// period), keeps the screen awake, and is the watchdog — an event that stalls is skipped (no points):
    /// no broadcast board after NoBoardSeconds, stuck in Ready, a heat live longer than MaxLiveSeconds, a result the
    /// HUD did not advance, repeated exceptions, or a scene older than MaxSceneSeconds.
    /// </summary>
    public class DemoRunner : MonoBehaviour
    {
        public const float GraceSeconds = 1.5f, NoBoardSeconds = 20f, MaxReadySeconds = 30f, MaxLiveSeconds = 180f,
                           ResultGraceSeconds = 15f, MaxSceneSeconds = 300f;
        const int MaxExceptions = 10;

        static DemoRunner _instance;
        float _sceneStart, _phaseStart;
        BoardPhase? _phase;
        int _exceptions;
        BroadcastHud _hud;

        /// <summary>One runner for the app's lifetime (created by the first demo, idle between demos).</summary>
        public static void Ensure()
        {
            if (_instance != null) { _instance.ResetScene(); return; }
            var go = new GameObject("DemoRunner");
            DontDestroyOnLoad(go);
            _instance = go.AddComponent<DemoRunner>();
        }

        void OnEnable()
        {
            SceneManager.sceneLoaded += OnSceneLoaded;
            Application.logMessageReceived += OnLog;
            ResetScene();
        }

        void OnDisable()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
            Application.logMessageReceived -= OnLog;
        }

        void OnSceneLoaded(Scene s, LoadSceneMode mode) => ResetScene();

        void ResetScene()
        {
            _sceneStart = _phaseStart = Time.unscaledTime;
            _phase = null;
            _exceptions = 0;
            _hud = null;
        }

        void OnLog(string msg, string stack, LogType type)
        {
            if (type == LogType.Exception) _exceptions++;
        }

        void Update()
        {
            if (!DemoMode.Active) return;             // idle until the next demo (no Destroy in runtime code)
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            float now = Time.unscaledTime, inScene = now - _sceneStart;
            if (inScene > GraceSeconds && AnyInput()) { DemoMode.Stop(); return; }
            if (SceneManager.GetActiveScene().name == MeetLineup.MenuScene) return;   // on the way back

            if (_hud == null) _hud = FindFirstObjectByType<BroadcastHud>();
            var board = _hud != null ? _hud.Board : null;
            if (board == null) { if (inScene > NoBoardSeconds) DemoMode.Advance("no event board"); return; }
            if (_phase != board.BoardState) { _phase = board.BoardState; _phaseStart = now; }
            float inPhase = now - _phaseStart;
            bool played = Gauntlet.CurrentHeatPlayed;   // after the scored heat the board may sit in Ready (held)
            if (_exceptions >= MaxExceptions) DemoMode.Advance($"{_exceptions} exceptions");
            else if (!played && _phase == BoardPhase.Ready && inPhase > MaxReadySeconds) DemoMode.Advance("stuck in Ready");
            else if (_phase == BoardPhase.Live && inPhase > MaxLiveSeconds) DemoMode.Advance($"heat over {MaxLiveSeconds:0} s");
            else if (played && now - _hud.ResultSince > DemoMode.ResultSeconds + ResultGraceSeconds) DemoMode.Advance("result not advanced");
            else if (inScene > MaxSceneSeconds) DemoMode.Advance($"scene over {MaxSceneSeconds:0} s");
        }

        static bool AnyInput() => AnyInput(held: false);

        /// <summary>Any mouse / touch / pen press, key or gamepad A on ANY device — not just Pointer.current (the
        /// Device Simulator's touchscreen is "current" while the mouse clicks the Game view).</summary>
        public static bool AnyInput(bool held)
        {
            foreach (var dev in InputSystem.devices)
            {
                if (dev is Pointer p && (held ? p.press.isPressed : p.press.wasPressedThisFrame)) return true;
                if (dev is Keyboard k && (held ? k.anyKey.isPressed : k.anyKey.wasPressedThisFrame)) return true;
                if (dev is Gamepad g && (held ? g.buttonSouth.isPressed : g.buttonSouth.wasPressedThisFrame)) return true;
            }
            return false;
        }
    }
}
