using System;
using System.Collections.Generic;
using System.Linq;
using PrimeTween;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// D3 broadcast HUD (UI Toolkit, 9:16): replaces the IMGUI placeholders for every 8-athlete event.
    ///   top bar      event title · subtitle · clock + info line
    ///   standings    place · lane (body chip) · live result · brain confidence sparkline + % (critic, feature 5) ·
    ///                win odds (from traits, Odds) · your bet
    ///   betting slip (optional, offerBets; off by default) each new heat waits in Ready (IBroadcastBoard.HoldStart)
    ///                until a bet is placed / skipped, or betWindowSeconds pass; stake Wallet.Stake coins, paid at the odds
    ///   banner       countdown / GO
    ///   ticker       play-by-play lines (Commentary: start, lead changes, athletes out, result)
    ///   stats card   the athlete the camera is on (BroadcastDirector.Subject: current winner / in trouble / winner):
    ///                speed + peak, power + peak, confidence (feature 7); refreshed once a second
    ///   result card  podium · bet outcome · world records (top 3, new-record highlight, the record to beat) · heat bests
    ///                · gauntlet points (Gauntlet) · New heat / Next / Menu
    /// Docked layout (user, 2026-09-29: "HUD on top and bottom so the gameplay is not covered"): the layout is authored
    /// at 1080 px wide and scaled onto the screen's safe area; the top dock (top bar + standings) and the bottom dock
    /// (stats card + ticker + buttons) are opaque panels, and the camera renders only into the gap between them
    /// (BroadcastCamera.SetViewport). Only the countdown banner and the modal overlays (betting slip, result card) sit
    /// over the game.
    /// One-screen portrait pass (GFX/UI idea 9, 2026-09-29): the standings show 4 rows (top 3 + the athlete in trouble /
    /// your bet; tap them for all 8), ODDS only with betting on, CONF only when a brain drops under 90 %, 28 px minimum
    /// text at 1080 wide, one action row (the result card no longer repeats "New heat"), 104 px buttons, and the safe
    /// area strips (status / navigation bar) filled with the dock colour instead of black.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public class BroadcastHud : MonoBehaviour
    {
        public MonoBehaviour board;             // an IBroadcastBoard
        public string title = "EVENT";
        public string subtitle = "Event";
        public int eventNumber;
        public string version = "v0";
        public TextAsset oddsModel;
        public StyleSheet style;
        [Tooltip("Betting slip before every heat (holds the start up to betWindowSeconds). Off by default (user, " +
                 "2026-09-29: \"just play\") — heats start straight away; the odds column stays as information.")]
        public bool offerBets;
        public float betWindowSeconds = 12f;
        [Tooltip("Feature 7 stats card + confidence column source (optional).")]
        public TensionMeter tension;
        public ArenaAudio arenaAudio;
        [Tooltip("Stadium screens (optional): world records and winners flash on them.")]
        public ScreenFeed screens;
        [Tooltip("Standings rows while collapsed (tap the standings for all).")]
        public int collapsedRows = 4;

        IBroadcastBoard B => board as IBroadcastBoard;
        Odds.Model _odds;
        readonly Commentary _pbp = new();
        VisualElement _root, _frame, _slip, _card, _rowsBox, _slipRows, _cardBody, _dockTop, _hole, _dockBottom, _safeTop, _safeBottom;
        Label _expandHint;
        Button _barNewHeat;
        bool _expanded;
        BroadcastCamera _viewCam;
        Label _title, _sub, _clock, _info, _banner, _ticker, _version, _coins, _slipTimer, _cardTitle;
        List<(int place, string name, string result, bool bad, PolicyRunner runner)> _frozen;   // gauntlet: the scored heat
        readonly List<(VisualElement row, Label place, Label chip, Label name, Label result, Sparkline spark, Label conf, Label odds)> _rows = new();
        VisualElement _stats;
        Label _statsChip, _statsName, _statsTag, _sSpeed, _sSpeedSub, _sPower, _sPowerSub, _sConf;
        float _statsNext;
        Sparkline _statsSpark;
        PolicyRunner _statsFor;
        BroadcastDirector _director;
        float _lastCall = -99f;
        float _heatTopSpeed, _liveSince;
        readonly Dictionary<PolicyRunner, float> _oddsBy = new();
        PolicyRunner _bet;
        float _betOdds;
        int _heatSeen = -1, _settledHeat = -1;
        float _heldFor;
        bool _slipOpen;
        const float DesignWidth = 1080f;

        void OnEnable()
        {
            _odds = Odds.Parse(oddsModel);
            var doc = GetComponent<UIDocument>();
            _root = doc.rootVisualElement;
            _root.Clear();
            if (style != null) _root.styleSheets.Add(style);
            _root.pickingMode = PickingMode.Ignore;
            _safeTop = Add(_root, "bh-safe");                 // status / navigation bar strips: dock colour, not black
            _safeBottom = Add(_root, "bh-safe");
            _frame = Add(_root, "bh-frame");
            _frame.pickingMode = PickingMode.Ignore;

            _dockTop = Add(_frame, "bh-dock-top");
            _hole = Add(_frame, "bh-hole");
            _hole.pickingMode = PickingMode.Ignore;
            _dockBottom = Add(_frame, "bh-dock-bottom");
            var top = Add(_dockTop, "bh-top");
            var titles = Add(top, "bh-titles");
            _title = AddLabel(titles, "bh-title", title);
            _sub = AddLabel(titles, "bh-sub", subtitle);
            var clockBox = Add(top, "bh-clockbox");
            _clock = AddLabel(clockBox, "bh-clock", "0.00 s");
            _info = AddLabel(clockBox, "bh-info", "");

            var standings = Add(_dockTop, "bh-standings");
            var head = Add(standings, "bh-row", "bh-head");
            AddLabel(head, "bh-c-place", "#");
            AddLabel(head, "bh-c-chip", "");
            AddLabel(head, "bh-c-name", "LANE");
            AddLabel(head, "bh-c-result", "RESULT");
            AddLabel(head, "bh-c-conf", "CONF");
            // the last column: ODDS with betting on, else the expand hint (rows keep an empty odds cell: aligned)
            if (offerBets) { AddLabel(head, "bh-c-odds", "ODDS"); _expandHint = new Label(); }
            else { _expandHint = AddLabel(head, "bh-c-odds", ""); _expandHint.AddToClassList("bh-expand"); }
            _rowsBox = Add(standings, "bh-rows");
            standings.RegisterCallback<ClickEvent>(_ => _expanded = !_expanded);

            _banner = AddLabel(_hole, "bh-banner", "");
            _banner.pickingMode = PickingMode.Ignore;
            BuildStatsCard();

            var bottom = Add(_dockBottom, "bh-bottom");
            _ticker = AddLabel(bottom, "bh-ticker", "");
            var bar = Add(bottom, "bh-bar");
            _barNewHeat = Button(bar, "New heat", () => { CloseSlip(); _card.style.display = DisplayStyle.None; B?.Restart(); }, "bh-btn-small");
            if (MeetLineup.MenuAvailable) Button(bar, "Menu", () => { Gauntlet.Abandon(); MeetLineup.ReturnToMenu(); }, "bh-btn-small");
            _coins = AddLabel(bar, "bh-coins", "");
            _version = AddLabel(bar, "bh-version", "ⓘ " + version.Split('·')[0].Trim());   // tap: performance overlay
            _version.tooltip = version;
            _version.RegisterCallback<ClickEvent>(_ => PerfOverlay.Toggle());

            // betting slip
            _slip = Add(_frame, "bh-overlay", "bh-slip");
            AddLabel(_slip, "bh-overlay-title", "PLACE YOUR BET");
            AddLabel(_slip, "bh-overlay-sub", $"{Wallet.Stake} coins on one athlete · paid at the odds if they win");
            _slipRows = Add(_slip, "bh-slip-rows");
            var slipBar = Add(_slip, "bh-slip-bar");
            Button(slipBar, "No bet — start", () => CloseSlip(), "bh-btn");
            _slipTimer = AddLabel(slipBar, "bh-slip-timer", "");
            _slip.style.display = DisplayStyle.None;

            // result card
            _card = Add(_frame, "bh-overlay", "bh-card");
            _cardTitle = AddLabel(_card, "bh-overlay-title", "RESULT");
            _cardBody = Add(_card, "bh-card-body");
            _card.style.display = DisplayStyle.None;          // its action is the bottom bar's New heat (one action row)
            if (tension != null) { tension.NearFall -= OnNearFall; tension.NearFall += OnNearFall; tension.Save -= OnSave; tension.Save += OnSave; }
        }

        static VisualElement Add(VisualElement parent, params string[] classes)
        {
            var v = new VisualElement();
            foreach (var c in classes) v.AddToClassList(c);
            parent.Add(v);
            return v;
        }

        static Label AddLabel(VisualElement parent, string cls, string text)
        {
            var l = new Label(text);
            l.AddToClassList(cls);
            parent.Add(l);
            return l;
        }

        static Button Button(VisualElement parent, string text, System.Action onClick, string cls)
        {
            var b = new Button(onClick) { text = text };
            b.AddToClassList(cls);
            parent.Add(b);
            return b;
        }

        void LateUpdate()
        {
            var b = B;
            if (b == null || _frame == null) return;
            FitToScreen();
            if (b.Heat != _heatSeen) NewHeat(b);
            var rows = b.Rows.ToList();
            bool stageDone = Gauntlet.Active && Gauntlet.CurrentHeatPlayed && _frozen != null;   // wait for Next ▸
            if (stageDone) rows = _frozen;
            else if (b.BoardState == BoardPhase.Ready) RefreshOdds(rows);
            HandleBetting(b, rows);
            _pbp.Update(b, Time.unscaledTime);

            _sub.text = $"{subtitle} · {b.SubtitleExtra}";
            _clock.text = b.ClockLine;
            _info.text = b.InfoLine;
            _banner.text = _slipOpen || stageDone || b.BoardState == BoardPhase.Result ? "" : b.Banner;
            _ticker.text = string.Join("\n", _pbp.Lines.AsEnumerable().Reverse().Take(2));
            _coins.text = offerBets ? $"{Wallet.Coins} coins" + (_bet != null ? $" · bet {Short(_bet, rows)} @ {Odds.Format(_betOdds)}" : "") : "";
            DrawRows(rows);
            UpdateStats(b, rows);
            _barNewHeat.EnableInClassList("bh-btn-primary", b.BoardState == BoardPhase.Result && !Gauntlet.Active);

            if (b.BoardState == BoardPhase.Result && _settledHeat != b.Heat) Settle(b, rows);
            if (b.BoardState != BoardPhase.Result && _card.style.display == DisplayStyle.Flex && !Gauntlet.Active) _card.style.display = DisplayStyle.None;
        }

        void NewHeat(IBroadcastBoard b)
        {
            _heatSeen = b.Heat;
            _bet = null;
            _heatTopSpeed = 0f;
            _heldFor = 0;
            _oddsBy.Clear();
            if (Gauntlet.Active && Gauntlet.CurrentHeatPlayed) { b.HoldStart = true; return; }   // one heat per gauntlet stage
            if (offerBets) { b.HoldStart = true; _slipOpen = true; _slip.style.display = DisplayStyle.Flex; _slipBuilt = false; }
        }

        bool _slipBuilt;

        void RefreshOdds(List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            var runners = rows.Select(r => r.runner).ToList();
            var p = Odds.WinProbabilities(_odds, eventNumber, runners);
            float margin = _odds?.margin ?? 0.1f;
            for (int i = 0; i < runners.Count; i++) _oddsBy[runners[i]] = Odds.Decimal(p[i], margin);
        }

        void HandleBetting(IBroadcastBoard b, List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            if (!_slipOpen) return;
            if (b.BoardState != BoardPhase.Ready) { CloseSlip(); return; }
            if (!_slipBuilt && _oddsBy.Count == rows.Count && rows.Count > 0)
            {
                _slipRows.Clear();
                foreach (var r in rows.OrderBy(r => r.name.Substring(1)))
                {
                    var runner = r.runner;
                    var btn = new Button(() => PlaceBet(runner)) { text = "" };
                    btn.AddToClassList("bh-slip-row");
                    var chip = AddLabel(btn, "bh-c-chip", Odds.IsZombie(runner) ? "Z" : "M");
                    chip.AddToClassList(Odds.IsZombie(runner) ? "bh-chip-zombie" : "bh-chip-matt");
                    AddLabel(btn, "bh-slip-name", r.name);
                    AddLabel(btn, "bh-slip-traits", $"STR {runner.strength:0.00} · LAT {runner.latencySubsteps} · NOISE {runner.obsNoise:0.00}");
                    AddLabel(btn, "bh-slip-odds", Odds.Format(_oddsBy[runner]));
                    _slipRows.Add(btn);
                }
                _slipBuilt = true;
            }
            _heldFor += Time.unscaledDeltaTime;
            float left = betWindowSeconds - _heldFor;
            _slipTimer.text = $"starts in {Mathf.CeilToInt(Mathf.Max(0, left))} s";
            if (left <= 0) CloseSlip();
        }

        void PlaceBet(PolicyRunner r)
        {
            Wallet.EnsureStake();
            Wallet.Coins -= Wallet.Stake;
            _bet = r;
            _betOdds = _oddsBy.TryGetValue(r, out var o) ? o : 1f;
            CloseSlip();
        }

        void CloseSlip()
        {
            _slipOpen = false;
            if (_slip != null) _slip.style.display = DisplayStyle.None;
            var b = B;
            if (b != null && !(Gauntlet.Active && Gauntlet.CurrentHeatPlayed)) b.HoldStart = false;
        }

        static string Short(PolicyRunner r, List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows) =>
            rows.FirstOrDefault(x => x.runner == r).name ?? "?";

        /// <summary>Collapsed standings: the top 3 + the story's 4th row (athlete in trouble, else your bet, else 4th).</summary>
        List<(int place, string name, string result, bool bad, PolicyRunner runner)> Shown(
            List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows, out List<int> ranks)
        {
            ranks = Enumerable.Range(0, rows.Count).ToList();
            int n = Mathf.Max(1, collapsedRows);
            _expandHint.text = rows.Count <= n ? "" : _expanded ? "TOP ▴" : $"ALL {rows.Count} ▾";
            if (_expanded || rows.Count <= n) return rows;
            var pick = Enumerable.Range(0, n - 1).ToList();
            var hot = tension != null && tension.HotDanger >= 0.45f ? tension.Hot : null;
            int extra = rows.FindIndex(r => r.runner != null && r.runner == hot && !r.bad);
            if (extra < n - 1) extra = rows.FindIndex(r => r.runner != null && r.runner == _bet);
            if (extra < n - 1) extra = n - 1;
            pick.Add(extra);
            ranks = pick;
            return pick.Select(i => rows[i]).ToList();
        }

        void DrawRows(List<(int place, string name, string result, bool bad, PolicyRunner runner)> all)
        {
            var rows = Shown(all, out var ranks);
            while (_rows.Count < rows.Count)
            {
                var row = Add(_rowsBox, "bh-row");
                var place = AddLabel(row, "bh-c-place", "");
                var chip = AddLabel(row, "bh-c-chip", "");
                var name = AddLabel(row, "bh-c-name", "");
                var result = AddLabel(row, "bh-c-result", "");
                var confBox = Add(row, "bh-c-conf");
                var spark = new Sparkline { capacity = 50, lineWidth = 2f };
                confBox.Add(spark);
                var conf = AddLabel(confBox, "bh-conf-pct", "");
                _rows.Add((row, place, chip, name, result, spark, conf, AddLabel(row, "bh-c-odds", "")));
            }
            for (int i = 0; i < _rows.Count; i++)
            {
                var ui = _rows[i];
                bool on = i < rows.Count;
                ui.row.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
                if (!on) continue;
                var r = rows[i];
                bool z = Odds.IsZombie(r.runner);
                ui.place.text = (r.place > 0 ? r.place : ranks[i] + 1).ToString();
                ui.chip.text = z ? "Z" : "M";
                ui.chip.EnableInClassList("bh-chip-zombie", z);
                ui.chip.EnableInClassList("bh-chip-matt", !z);
                ui.name.text = r.name;
                ui.result.text = r.result;
                ui.odds.text = offerBets && _oddsBy.TryGetValue(r.runner, out var o) ? Odds.Format(o) : "";
                var tel = Telemetry(r.runner);
                // confidence only when it tells something: a brain under 90 % (a column of "100%" is noise)
                bool hasConf = tel != null && !float.IsNaN(tel.Confidence) && !r.bad && tel.Confidence < 0.9f;
                ui.conf.text = hasConf ? $"{tel.Confidence * 100f:0}%" : r.bad ? "—" : "";
                ui.conf.style.color = hasConf ? Sparkline.ColorOf(tel.Confidence) : new Color(0.55f, 0.6f, 0.69f);
                if (tel != null) ui.spark.SetValues(tel.ConfidenceHistory);
                ui.spark.style.visibility = hasConf ? Visibility.Visible : Visibility.Hidden;
                ui.row.EnableInClassList("bh-row-hot", tension != null && tension.Hot == r.runner && tension.HotDanger >= 0.45f && !r.bad);
                ui.row.EnableInClassList("bh-row-bad", r.bad);
                ui.row.EnableInClassList("bh-row-first", r.place == 1);
                ui.row.EnableInClassList("bh-row-bet", r.runner == _bet);
            }
        }

        void Settle(IBroadcastBoard b, List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            _settledHeat = b.Heat;
            _cardBody.Clear();
            var podium = rows.Where(r => r.place >= 1 && r.place <= 3).OrderBy(r => r.place).ToList();
            var winner = podium.FirstOrDefault(r => r.place == 1);
            _cardTitle.text = winner.name != null ? $"{winner.name} WINS" : "RESULT";
            string[] medal = { "", "gold", "silver", "bronze" };
            foreach (var r in podium)
            {
                var line = Add(_cardBody, "bh-podium", "bh-podium-" + medal[Mathf.Clamp(r.place, 1, 3)]);
                AddLabel(line, "bh-podium-place", r.place.ToString());
                var chip = AddLabel(line, "bh-c-chip", Odds.IsZombie(r.runner) ? "Z" : "M");
                chip.AddToClassList(Odds.IsZombie(r.runner) ? "bh-chip-zombie" : "bh-chip-matt");
                AddLabel(line, "bh-podium-name", r.name);
                AddLabel(line, "bh-podium-result", r.result);
            }
            // bet
            if (_bet != null)
            {
                bool won = rows.Any(r => r.runner == _bet && r.place == 1);
                int pay = won ? Mathf.RoundToInt(Wallet.Stake * _betOdds) : 0;
                if (won) Wallet.Coins += pay;
                AddLabel(_cardBody, won ? "bh-card-good" : "bh-card-line",
                    won ? $"Your bet on {Short(_bet, rows)} @ {Odds.Format(_betOdds)} pays {pay} coins!" : $"Your bet on {Short(_bet, rows)} lost ({Wallet.Stake} coins).");
                _pbp.Add(won ? $"Bet won: +{pay} coins" : "Bet lost");
            }
            // world records
            if (eventNumber > 0) WorldRecords(b, winner);
            HeatBests(rows);
            if (winner.name != null) _pbp.Add($"{winner.name} wins — {winner.result}");
            if (winner.name != null && screens != null) screens.Banner($"{winner.name} WINS", 4f);
            // gauntlet
            if (Gauntlet.Active && !Gauntlet.CurrentHeatPlayed)
            {
                Gauntlet.Award(rows.Select(r => (r.name, r.place)));
                _frozen = rows;
                AddLabel(_cardBody, "bh-card-line", $"GAUNTLET · event {Gauntlet.Index + 1} of {Gauntlet.Events.Length}");
                foreach (var (lane, pts) in Gauntlet.Table().Take(4)) AddLabel(_cardBody, "bh-card-line", $"{lane}   {pts} pts");
                var next = new Button(() => Gauntlet.Next()) { text = Gauntlet.IsLast ? "Final standings" : "Next event ▸" };
                next.AddToClassList("bh-btn");
                _cardBody.Add(next);
            }
            _card.style.display = DisplayStyle.Flex;
        }

        AthleteTelemetry Telemetry(PolicyRunner r) =>
            r == null ? null : tension != null ? tension.TelemetryOf(r) : r.GetComponent<AthleteTelemetry>();

        static string Body(PolicyRunner r) => r == null ? "" : Odds.IsZombie(r) ? "ZOMBIE" : "MATT";

        /// <summary>Result card: this heat's mark against the world records (top 3, the new entry highlighted).</summary>
        void WorldRecords(IBroadcastBoard b, (int place, string name, string result, bool bad, PolicyRunner runner) winner)
        {
            var box = Add(_cardBody, "bh-wr");
            var head = Add(box, "bh-wr-head");
            AddLabel(head, "bh-wr-title", "WORLD RECORDS");
            int rank = 0;
            double gap = double.NaN;
            bool has = b.TryWinningMark(out var v, out var lower, out var text);
            if (has)
            {
                gap = Records.GapToRecord(eventNumber, v);
                rank = Records.Submit(eventNumber, v, lower, text, winner.name ?? "", Body(winner.runner), b.Heat + 1);
            }
            var tag = AddLabel(head, "bh-wr-tag", rank == 1 ? "NEW WORLD RECORD!" : rank > 1 ? $"#{rank} ALL-TIME" : "");
            var top = Records.Top(eventNumber);
            for (int i = 0; i < Mathf.Min(3, top.Count); i++)
            {
                var e = top[i];
                var line = Add(box, "bh-wr-row");
                line.EnableInClassList("bh-wr-row-new", i + 1 == rank);
                AddLabel(line, "bh-wr-rank", (i + 1).ToString());
                AddLabel(line, "bh-wr-mark", e.mark);
                AddLabel(line, "bh-wr-who", e.holder + (string.IsNullOrEmpty(e.body) ? "" : $" · {e.body}"));
                AddLabel(line, "bh-wr-date", e.date ?? "");
            }
            if (top.Count == 0) AddLabel(box, "bh-card-line", "No mark yet: the first valid winning mark sets it.");
            else if (has && rank != 1 && !double.IsNaN(gap))
                AddLabel(box, "bh-card-line", $"{winner.name} {text} · world record {top[0].mark}");
            if (rank == 1)
            {
                _pbp.Add($"WORLD RECORD! {winner.name} — {text}");
                if (arenaAudio != null) arenaAudio.NewRecord();
                if (screens != null) screens.Banner("WORLD RECORD", 7f);
                Tween.Custom(1f, 1.08f, 0.35f, s => tag.style.scale = new Scale(new Vector3(s, s, 1f)), ease: Ease.InOutSine,
                             cycles: 6, cycleMode: CycleMode.Yoyo, useUnscaledTime: true);
            }
            else if (rank > 1) _pbp.Add($"{winner.name} goes #{rank} on the all-time list");
            Tween.Custom(0f, 1f, 0.45f, a => box.style.opacity = a, ease: Ease.OutQuad, startDelay: 0.25f, useUnscaledTime: true);
        }

        /// <summary>Result card: the heat's physical bests from telemetry (fastest, most powerful, closest call).</summary>
        void HeatBests(List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            var tel = rows.Select(r => (r.name, r.bad, t: Telemetry(r.runner))).Where(x => x.t != null && x.t.Ready).ToList();
            if (tel.Count == 0) return;
            var parts = new List<string>();
            var fast = tel.OrderByDescending(x => x.t.PeakSpeedMps).First();
            bool race = tension == null || tension.kind == BroadcastDirector.Kind.Race;       // stationary events: a speed is a fall
            if (race) parts.Add($"top speed {fast.t.PeakSpeedMps:0.0} m/s {fast.name}");
            var strong = tel.OrderByDescending(x => x.t.PeakPowerW).First();
            parts.Add($"peak power {strong.t.PeakPowerW:N0} W {strong.name}");
            // closest call = the lowest confidence of an athlete who stayed in (a faller's 0 % is not a "call")
            var close = tel.Where(x => !x.bad && !float.IsNaN(x.t.Confidence)).OrderBy(x => x.t.MinConfidence).FirstOrDefault();
            if (close.t != null && close.t.MinConfidence < 0.5f) parts.Add($"closest call {close.name} ({close.t.MinConfidence * 100f:0}%)");
            AddLabel(_cardBody, "bh-card-line", "Heat bests · " + string.Join(" · ", parts));
        }

        void BuildStatsCard()
        {
            _stats = Add(_dockBottom, "bh-stats");
            _stats.pickingMode = PickingMode.Ignore;
            var head = Add(_stats, "bh-stats-head");
            _statsChip = AddLabel(head, "bh-c-chip", "");
            _statsName = AddLabel(head, "bh-stats-name", "");
            _statsTag = AddLabel(head, "bh-stats-tag", "");
            var grid = Add(_stats, "bh-stats-grid");
            (Label v, Label sub) Tile(string title)
            {
                var t = Add(grid, "bh-tile");
                AddLabel(t, "bh-tile-title", title);
                return (AddLabel(t, "bh-tile-value", "—"), AddLabel(t, "bh-tile-sub", ""));
            }
            // simple card (user, 2026-09-29: "simplify this … no more than once a second"): speed, power, confidence
            (_sSpeed, _sSpeedSub) = Tile("SPEED");
            (_sPower, _sPowerSub) = Tile("POWER");
            var conf = Add(grid, "bh-tile", "bh-tile-conf");
            AddLabel(conf, "bh-tile-title", "CONFIDENCE");
            var row = Add(conf, "bh-tile-conf-row");
            _sConf = AddLabel(row, "bh-tile-value", "—");
            _statsSpark = new Sparkline { capacity = 60 };
            _statsSpark.AddToClassList("bh-stats-spark");
            row.Add(_statsSpark);
            _stats.style.visibility = Visibility.Hidden;
        }

        void UpdateStats(IBroadcastBoard b, List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            if (_stats == null) return;
            if (_director == null) _director = FindAnyObjectByType<BroadcastDirector>();
            // the card follows the camera's story (the athlete on screen) and refreshes once a second: no flicker
            if (Time.unscaledTime >= _statsNext || _slipOpen)
            {
                _statsNext = Time.unscaledTime + 1f;
                PolicyRunner who;
                string tag;
                if (b.BoardState == BoardPhase.Result) { who = rows.FirstOrDefault(r => r.place == 1).runner; tag = "WINNER"; }
                else
                {
                    who = _director != null && _director.Subject != null ? _director.Subject
                        : tension != null && tension.Leader != null ? tension.Leader : rows.FirstOrDefault(r => !r.bad).runner;
                    tag = _director != null && _director.Current == BroadcastDirector.Shot.Hot ? "IN TROUBLE"
                        : b.BoardState == BoardPhase.Live ? "LEADER" : "ON THE LINE";
                }
                var t = Telemetry(who);
                bool show = t != null && t.Ready && !_slipOpen && rows.Any(r => r.runner == who);
                _stats.style.visibility = show ? Visibility.Visible : Visibility.Hidden;
                if (!show) _statsFor = null;
                else
                {
                    if (who != _statsFor)
                    {
                        _statsFor = who;
                        Tween.Custom(0.4f, 1f, 0.3f, a => _stats.style.opacity = a, useUnscaledTime: true);
                    }
                    var row = rows.First(r => r.runner == who);
                    bool z = Odds.IsZombie(who);
                    _statsChip.text = z ? "Z" : "M";
                    _statsChip.EnableInClassList("bh-chip-zombie", z);
                    _statsChip.EnableInClassList("bh-chip-matt", !z);
                    _statsName.text = $"{row.name}  {Body(who)}";
                    _statsTag.text = tag;
                    _statsTag.EnableInClassList("bh-stats-tag-hot", tag == "IN TROUBLE");
                    _sSpeed.text = $"{t.SpeedMps:0.0} m/s";
                    _sSpeedSub.text = $"peak {t.PeakSpeedMps:0.0}";
                    _sPower.text = $"{t.PowerW:N0} W";
                    _sPowerSub.text = $"peak {t.PeakPowerW:N0}";
                    bool hasConf = !float.IsNaN(t.Confidence);
                    _sConf.text = hasConf ? $"{t.Confidence * 100f:0}%" : "—";
                    _sConf.style.color = hasConf ? Sparkline.ColorOf(t.Confidence) : new Color(0.55f, 0.6f, 0.69f);
                    _statsSpark.SetValues(t.ConfidenceHistory);
                }
            }

            // play-by-play from the telemetry: a new heat top speed (runs above 2.5 m/s), with a cooldown
            if (b.BoardState != BoardPhase.Live) _liveSince = Time.unscaledTime;
            if (b.BoardState == BoardPhase.Live && Time.unscaledTime - _liveSince > 4f && Time.unscaledTime - _lastCall > 6f)
            {
                var fastest = rows.Where(r => !r.bad).Select(r => (r.name, t: Telemetry(r.runner))).Where(x => x.t != null && x.t.Ready)
                                  .OrderByDescending(x => x.t.PeakSpeedMps).FirstOrDefault();
                if (fastest.t != null && fastest.t.PeakSpeedMps > 2.5f && fastest.t.PeakSpeedMps > _heatTopSpeed + 0.15f)
                {
                    if (_heatTopSpeed > 0f) { _pbp.Add($"{fastest.name} hits {fastest.t.PeakSpeedMps:0.0} m/s, fastest of the heat"); _lastCall = Time.unscaledTime; }
                    _heatTopSpeed = fastest.t.PeakSpeedMps;
                }
            }
        }

        void OnNearFall(PolicyRunner r)
        {
            var t = Telemetry(r);
            string name = NameOf(r);
            if (name == null) return;
            _pbp.Add(t != null && !float.IsNaN(t.Confidence) ? $"{name} is wobbling — brain confidence {t.Confidence * 100f:0}%!" : $"{name} is wobbling!");
        }

        void OnSave(PolicyRunner r) { var n = NameOf(r); if (n != null) _pbp.Add($"What a save by {n}!"); }

        string NameOf(PolicyRunner r) => B?.Rows.FirstOrDefault(x => x.runner == r).name;

        void OnDisable()
        {
            if (tension != null) { tension.NearFall -= OnNearFall; tension.Save -= OnSave; }
        }

        /// <summary>Scale the 1080-wide layout onto the screen's safe area and give the camera the gap between the
        /// docks (in screen pixels → normalised viewport).</summary>
        void FitToScreen()
        {
            var panel = _root.panel;
            if (panel == null) return;
            var safe = Screen.safeArea;
            var tl = RuntimePanelUtils.ScreenToPanel(panel, new Vector2(safe.xMin, Screen.height - safe.yMax));
            var br = RuntimePanelUtils.ScreenToPanel(panel, new Vector2(safe.xMax, Screen.height - safe.yMin));
            float w = br.x - tl.x, h = br.y - tl.y;
            if (w <= 1 || h <= 1) return;
            float s = w / DesignWidth;
            var full = panel.visualTree.layout;
            _safeTop.style.left = 0; _safeTop.style.right = 0; _safeTop.style.top = 0; _safeTop.style.height = Mathf.Max(0f, tl.y);
            _safeBottom.style.left = 0; _safeBottom.style.right = 0; _safeBottom.style.top = br.y;
            _safeBottom.style.height = float.IsNaN(full.height) ? 0f : Mathf.Max(0f, full.height - br.y);
            _frame.style.left = tl.x;
            _frame.style.top = tl.y;
            _frame.style.width = DesignWidth;
            _frame.style.height = h / s;
            _frame.style.transformOrigin = new TransformOrigin(0, 0);
            _frame.style.scale = new Scale(new Vector3(s, s, 1));

            float topH = _dockTop.layout.height, botH = _dockBottom.layout.height;
            if (float.IsNaN(topH) || float.IsNaN(botH) || topH <= 0f) return;      // first frame: not laid out yet
            float px = safe.height / h * s;                                          // design units → screen pixels
            float y0 = safe.yMin + botH * px, y1 = safe.yMax - topH * px;
            if (y1 - y0 < 64f) return;
            if (_viewCam == null && Camera.main != null) _viewCam = Camera.main.GetComponent<BroadcastCamera>();
            if (_viewCam != null)
                _viewCam.SetViewport(new Rect(safe.xMin / Screen.width, y0 / Screen.height, safe.width / Screen.width, (y1 - y0) / Screen.height));
        }

        void OnDestroy() { if (_viewCam != null) _viewCam.ClearViewport(); }
    }
}
