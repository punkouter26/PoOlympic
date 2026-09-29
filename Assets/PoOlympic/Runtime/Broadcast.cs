using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>What an 8-athlete event shows on the broadcast HUD.</summary>
    public interface IStandingsBoard
    {
        string SubtitleExtra { get; }
        string ClockLine { get; }
        string InfoLine { get; }
        /// <summary>In display order: place (0 while live), name, result text, highlight (out / DQ), the athlete.</summary>
        IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows { get; }
        /// <summary>Countdown / GO / winner text, empty for none.</summary>
        string Banner { get; }
        void Restart();
    }

    public enum BoardPhase { Ready, Live, Result }

    /// <summary>Game-layer view of an event (D3 broadcast + betting): phase, heat counter, a start hold for the betting
    /// window, and the winner's mark for the records table.</summary>
    public interface IBroadcastBoard : IStandingsBoard
    {
        BoardPhase BoardState { get; }
        int Heat { get; }
        /// <summary>While true the event stays in Ready with its countdown reset (betting window).</summary>
        bool HoldStart { get; set; }
        /// <summary>Result phase only: the winner's mark (false when the heat has no valid mark, e.g. everyone DQ).</summary>
        bool TryWinningMark(out double value, out bool lowerIsBetter, out string text);
    }

    /// <summary>
    /// Betting odds from athlete traits (tasks.md D4). Per event a Plackett-Luce rating fitted on CPU heats
    /// (training/tools/fit_odds.py → Models/odds_model.json):
    ///   rating = w · [zombie, strength − 1, latency substeps, obs noise],  P(win) = softmax(rating)
    /// P is shrunk 10 % towards uniform; decimal odds = (1 − margin) / P, clamped to [1.01, 50].
    /// </summary>
    public static class Odds
    {
        [Serializable] public class EventWeights { public int number; public string name; public float[] weights; public int heats; public float favourite_wins; }
        [Serializable] public class Model { public float margin = 0.1f; public EventWeights[] events; }

        public static Model Parse(TextAsset json) => json == null ? null : JsonUtility.FromJson<Model>(json.text);

        public static bool IsZombie(PolicyRunner r) => r.athletePrefix.StartsWith("Z") || (r.contractJson != null && r.contractJson.name.Contains("zombie"));

        public static float Rating(float[] w, PolicyRunner r) =>
            w[0] * (IsZombie(r) ? 1f : 0f) + w[1] * (r.strength - 1f) + w[2] * r.latencySubsteps + w[3] * r.obsNoise;

        /// <summary>Win probability per runner (same order); uniform when the event has no fitted weights.</summary>
        public static float[] WinProbabilities(Model model, int eventNumber, IReadOnlyList<PolicyRunner> runners)
        {
            var w = model?.events?.FirstOrDefault(e => e.number == eventNumber)?.weights;
            var p = new float[runners.Count];
            if (w == null || w.Length < 4)
            {
                for (int i = 0; i < p.Length; i++) p[i] = 1f / p.Length;
                return p;
            }
            var r = runners.Select(x => Rating(w, x)).ToArray();
            float m = r.Max(), z = 0;
            for (int i = 0; i < p.Length; i++) z += p[i] = Mathf.Exp(r[i] - m);
            // shrink towards uniform: the fit is in-sample on 60 heats per event, so no athlete is ever a sure loss
            for (int i = 0; i < p.Length; i++) p[i] = (1f - Shrink) * p[i] / z + Shrink / p.Length;
            return p;
        }

        public const float Shrink = 0.1f, MaxOdds = 50f;

        public static float Decimal(float p, float margin) =>
            Mathf.Clamp(Mathf.Round((1f - margin) / Mathf.Max(p, 1e-4f) * 20f) / 20f, 1.01f, MaxOdds);

        public static string Format(float odds) => odds < 10f ? odds.ToString("0.00") : odds.ToString("0.0");
    }

    /// <summary>Virtual coins for the betting slip (PlayerPrefs: per device, never real money).</summary>
    public static class Wallet
    {
        const string Key = "poolympic.wallet";
        public const int StartCoins = 100, Stake = 10;
        public static int Coins
        {
            get => PlayerPrefs.GetInt(Key, StartCoins);
            set { PlayerPrefs.SetInt(Key, Mathf.Max(0, value)); PlayerPrefs.Save(); }
        }
        /// <summary>Broke: top up so the game can always go on.</summary>
        public static void EnsureStake() { if (Coins < Stake) Coins = StartCoins; }
    }

    /// <summary>
    /// World records: the best winning marks per event (top <see cref="Keep"/>, PlayerPrefs JSON), each with the mark
    /// text, holder (lane label), body, heat and date. Rank 1 = the world record. The pre-list single record
    /// (poolympic.record.NN = "value|text · who") is migrated on first read.
    /// </summary>
    public static class Records
    {
        public const int Keep = 5;

        [Serializable] public class Entry { public double value; public string mark; public string holder; public string body; public string date; public int heat; }
        [Serializable] class Table { public bool lowerIsBetter; public List<Entry> entries = new(); }

        static string Key(int ev) => $"poolympic.records.{ev:00}";
        static string OldKey(int ev) => $"poolympic.record.{ev:00}";

        static Table Load(int ev)
        {
            var json = PlayerPrefs.GetString(Key(ev), "");
            if (json.Length > 0)
            {
                try { return JsonUtility.FromJson<Table>(json) ?? new Table(); } catch (ArgumentException) { return new Table(); }
            }
            var t = new Table();
            var s = PlayerPrefs.GetString(OldKey(ev), "");
            int bar = s.IndexOf('|');
            if (bar > 0 && double.TryParse(s.Substring(0, bar), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v))
            {
                var rest = s.Substring(bar + 1);
                int dot = rest.LastIndexOf(" · ", StringComparison.Ordinal);
                t.entries.Add(new Entry { value = v, mark = dot > 0 ? rest.Substring(0, dot) : rest, holder = dot > 0 ? rest.Substring(dot + 3) : "", body = "", date = "" });
            }
            return t;
        }

        static void Save(int ev, Table t)
        {
            PlayerPrefs.SetString(Key(ev), JsonUtility.ToJson(t));
            PlayerPrefs.Save();
        }

        /// <summary>Best first; empty when the event has no record yet.</summary>
        public static IReadOnlyList<Entry> Top(int ev) => Load(ev).entries;

        /// <summary>The world record: "mark · holder" text.</summary>
        public static bool TryGet(int ev, out double value, out string text)
        {
            var top = Load(ev).entries;
            value = 0; text = "";
            if (top.Count == 0) return false;
            value = top[0].value;
            text = Describe(top[0]);
            return true;
        }

        public static string Describe(Entry e) =>
            e.mark + (string.IsNullOrEmpty(e.holder) ? "" : $" · {e.holder}") + (string.IsNullOrEmpty(e.body) ? "" : $" ({e.body})");

        /// <summary>Stores the mark if it makes the top <see cref="Keep"/>. Returns its rank (1 = new world record) or 0.</summary>
        public static int Submit(int ev, double value, bool lowerIsBetter, string mark, string holder, string body = "", int heat = 0)
        {
            var t = Load(ev);
            t.lowerIsBetter = lowerIsBetter;
            int rank = t.entries.Count(e => lowerIsBetter ? e.value <= value : e.value >= value) + 1;   // ties: the older mark stays ahead
            if (rank > Keep) return 0;
            t.entries.Insert(rank - 1, new Entry { value = value, mark = mark, holder = holder, body = body, heat = heat,
                                                    date = DateTime.Now.ToString("d MMM yyyy", System.Globalization.CultureInfo.InvariantCulture) });
            while (t.entries.Count > Keep) t.entries.RemoveAt(t.entries.Count - 1);
            Save(ev, t);
            return rank;
        }

        /// <summary>Signed gap of a mark to the world record in the mark's own unit, "better" = negative for lower-is-better
        /// events. NaN without a record.</summary>
        public static double GapToRecord(int ev, double value)
        {
            var top = Load(ev).entries;
            return top.Count == 0 ? double.NaN : value - top[0].value;
        }

        public static void Clear(int ev) { PlayerPrefs.DeleteKey(Key(ev)); PlayerPrefs.DeleteKey(OldKey(ev)); PlayerPrefs.Save(); }
    }

    /// <summary>
    /// Play-by-play ticker lines from a board, by diffing its rows frame to frame (works for every event without
    /// per-event hooks): start, lead changes (debounced), athletes going out, the result.
    /// </summary>
    public sealed class Commentary
    {
        public readonly List<string> Lines = new();
        public int MaxLines = 4;
        readonly Dictionary<string, bool> _bad = new();
        BoardPhase _phase = BoardPhase.Ready;
        int _heat = -1;
        string _leader = "", _pending = "";
        float _pendingSince, _lastLeadCall = -99f;
        public float LeadCooldown = 4f;                  // s between lead-change calls (neck-and-neck races flip often)

        public void Add(string line)
        {
            Lines.Add(line);
            while (Lines.Count > MaxLines) Lines.RemoveAt(0);
        }

        public void Update(IBroadcastBoard b, float time)
        {
            if (b.Heat != _heat) { _heat = b.Heat; _bad.Clear(); _leader = _pending = ""; Add($"Heat {b.Heat + 1} — athletes to their marks"); }
            var rows = b.Rows.ToList();
            if (b.BoardState != _phase)
            {
                if (b.BoardState == BoardPhase.Live) Add("And they're off!");
                _phase = b.BoardState;
            }
            if (b.BoardState == BoardPhase.Live && rows.Count > 0)
            {
                var top = rows.FirstOrDefault(r => !r.bad).name ?? "";
                if (top != _leader)
                {
                    if (top != _pending) { _pending = top; _pendingSince = time; }
                    else if (time - _pendingSince > 1.2f && top.Length > 0 && time - _lastLeadCall > LeadCooldown)
                    {
                        if (_leader.Length > 0) { Add($"{top} takes the lead from {_leader}"); _lastLeadCall = time; }
                        _leader = top;
                    }
                }
            }
            foreach (var r in rows)
            {
                _bad.TryGetValue(r.name, out var was);
                if (r.bad && !was && b.BoardState != BoardPhase.Ready) Add($"{r.name} is out — {r.result}");
                _bad[r.name] = r.bad;
            }
        }
    }
}
