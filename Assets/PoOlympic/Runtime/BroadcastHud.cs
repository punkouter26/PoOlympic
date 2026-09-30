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
    /// Docked layout (user, 2026-09-29: "HUD on top and bottom so the gameplay is not covered"): the layout is authored
    /// at 1080 px wide and scaled onto the screen's safe area; the top and bottom docks are opaque panels and the camera
    /// renders only into the gap between them (BroadcastCamera.SetViewport). Only the countdown / winner banner, the
    /// menu sheet, the debug readout and the optional betting slip sit over the game.
    /// Consolidated portrait layout (UI consolidation, 2026-09-29 — one viewport, nothing scrolls, no result card):
    ///   top dock     HudAnchors frame: TL event title + subtitle · TC FPS · TR clock + info line, menu sheet (New heat,
    ///                Main menu, all 8 / top 4, performance)
    ///                standings: place · lane (body chip) · live result · badge (WR / BET) · brain confidence sparkline
    ///                + % (only under 90 %) · odds (betting on) / the "8 ▾" expand hint; 4 rows (top 3 + the story:
    ///                athlete in trouble / your bet), tap for all 8
    ///                result (in place of the old result card): rows 1-3 in gold / silver / bronze, the world-record
    ///                line (new record pulses, else the record to beat), heat bests, bet outcome, gauntlet points
    ///   bottom dock  story strip: the athlete the camera is on (BroadcastDirector.Subject: leader / in trouble / winner)
    ///                with speed · power · confidence, refreshed once a second (feature 7)
    ///                HudAnchors bottom bar: BL debug · play-by-play ticker (Commentary), which gives way to the primary
    ///                action at the result (New heat, or Next event in a gauntlet) · BR version
    ///   betting slip (optional, offerBets; off by default) each new heat waits in Ready (IBroadcastBoard.HoldStart)
    ///                until a bet is placed / skipped, or betWindowSeconds pass; stake Wallet.Stake coins, paid at the odds
    /// Rules (UI/Tokens.uss): no text under 28 px at 1080 wide, 104 px primary buttons; HudLayoutAudit + the EditMode
    /// layout tests check them at several phone sizes.
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
        [Tooltip("Story strip + confidence column source (optional).")]
        public TensionMeter tension;
        public ArenaAudio arenaAudio;
        [Tooltip("Stadium screens (optional): world records and winners flash on them.")]
        public ScreenFeed screens;
        [Tooltip("Standings rows while collapsed (tap the standings for all).")]
        public int collapsedRows = 4;
        [Tooltip("Seconds the winner banner stays over the game at the result.")]
        public float winnerBannerSeconds = 2.5f;

        /// <summary>Tests / previews: a board that is not a scene component.</summary>
        public IBroadcastBoard BoardOverride { get; set; }
        IBroadcastBoard B => BoardOverride ?? board as IBroadcastBoard;

        public VisualElement Frame => _frame;
        public VisualElement Hole => _hole;
        public HudAnchors Anchors => _anchors;
        public bool Expanded { get => _expanded; set => _expanded = value; }

        Odds.Model _odds;
        readonly Commentary _pbp = new();
        HudAnchors _anchors;
        VisualElement _root, _frame, _slip, _rowsBox, _slipRows, _dockTop, _hole, _dockBottom, _safeTop, _safeBottom, _resultBox;
        Button _primary;
        bool _expanded;
        BroadcastCamera _viewCam;
        Label _clock, _info, _banner, _ticker, _coins, _slipTimer;
        List<(int place, string name, string result, bool bad, PolicyRunner runner)> _frozen;   // gauntlet: the scored heat
        readonly List<(VisualElement row, Label place, Label chip, Label name, Label badge, Label result, Sparkline spark, Label conf, Label odds)> _rows = new();
        VisualElement _story;
        Label _storyChip, _storyName, _storyTag, _sSpeed, _sPower, _sConf;
        float _storyNext;
        BoardPhase _storyPhase;
        Sparkline _storySpark;
        PolicyRunner _storyFor;
        BroadcastDirector _director;
        float _lastCall = -99f;
        float _heatTopSpeed, _liveSince, _resultSince = -99f;
        readonly Dictionary<PolicyRunner, float> _oddsBy = new();
        PolicyRunner _bet, _recordHolder;
        float _betOdds;
        int _heatSeen = -1, _settledHeat = -1;
        float _heldFor;
        bool _slipOpen;
        const float DesignWidth = 1080f;

        void OnEnable() => Build();

        /// <summary>Build the element tree (OnEnable; tests call it directly in edit mode).</summary>
        public void Build()
        {
            _odds = Odds.Parse(oddsModel);
            var doc = GetComponent<UIDocument>();
            _root = doc.rootVisualElement;
            _root.Clear();
            _rows.Clear();
            _heatSeen = _settledHeat = -1;
            if (style != null) _root.styleSheets.Add(style);
            _root.AddToClassList("po-tokens");
            _root.pickingMode = PickingMode.Ignore;
            _safeTop = Add(_root, "bh-safe");                 // status / navigation bar strips: dock colour, not black
            _safeBottom = Add(_root, "bh-safe");
            _frame = Add(_root, "bh-frame");
            _frame.pickingMode = PickingMode.Ignore;

            _dockTop = Add(_frame, "bh-dock-top");
            _hole = Add(_frame, "bh-hole");
            _hole.pickingMode = PickingMode.Ignore;
            _dockBottom = Add(_frame, "bh-dock-bottom");

            _banner = AddLabel(_hole, "bh-banner", "");
            _banner.pickingMode = PickingMode.Ignore;

            // anchors: TL title · TC fps · TR clock + menu · BL debug · BR version
            _anchors = new HudAnchors(_dockTop, _dockBottom, _frame, _hole, title, subtitle, version.Split('·')[0].Trim());
            _anchors.Version.tooltip = version;
            var clockBox = Add(_anchors.Extra, "bh-clockbox");
            _clock = AddLabel(clockBox, "bh-clock", "0.00 s");
            _info = AddLabel(clockBox, "bh-info", "");
            _anchors.SetMenu(MenuItems);

            var standings = Add(_dockTop, "bh-standings");
            _rowsBox = Add(standings, "bh-rows");
            standings.RegisterCallback<ClickEvent>(_ => _expanded = !_expanded);
            _resultBox = Add(standings, "bh-result");
            _resultBox.style.display = DisplayStyle.None;

            BuildStory();
            _coins = AddLabel(_anchors.BottomCentre, "bh-coins", "");
            _coins.style.display = offerBets ? DisplayStyle.Flex : DisplayStyle.None;
            _ticker = AddLabel(_anchors.BottomCentre, "bh-ticker", "");
            _primary = new Button(PrimaryAction) { text = "New heat" };
            _primary.AddToClassList("bh-btn-small");
            _primary.AddToClassList("bh-btn-primary");
            _primary.style.display = DisplayStyle.None;
            _anchors.BottomCentre.Add(_primary);

            // betting slip
            _slip = Add(_frame, "bh-overlay", "bh-slip");
            AddLabel(_slip, "bh-overlay-title", "PLACE YOUR BET");
            AddLabel(_slip, "bh-overlay-sub", $"{Wallet.Stake} coins on one athlete · paid at the odds if they win");
            _slipRows = Add(_slip, "bh-slip-rows");
            var slipBar = Add(_slip, "bh-slip-bar");
            Button(slipBar, "No bet — start", () => CloseSlip(), "bh-btn");
            _slipTimer = AddLabel(slipBar, "bh-slip-timer", "");
            _slip.style.display = DisplayStyle.None;

            if (tension != null) { tension.NearFall -= OnNearFall; tension.NearFall += OnNearFall; tension.Save -= OnSave; tension.Save += OnSave; }
        }

        IEnumerable<(string, Action)> MenuItems()
        {
            yield return ("New heat", Restart);
            if (MeetLineup.MenuAvailable) yield return (Gauntlet.Active ? "Main menu (ends the gauntlet)" : "Main menu", () => { Gauntlet.Abandon(); MeetLineup.ReturnToMenu(); });
            yield return (_expanded ? "Standings: top 4" : "Standings: all 8", () => _expanded = !_expanded);
            yield return (HudAnchors.DebugOn ? "Performance: hide" : "Performance: show", () => HudAnchors.DebugOn = !HudAnchors.DebugOn);
        }

        void Restart()
        {
            CloseSlip();
            B?.Restart();
        }

        void PrimaryAction()
        {
            if (Gauntlet.Active && Gauntlet.CurrentHeatPlayed) Gauntlet.Next();
            else Restart();
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

        static bool Zombie(PolicyRunner r) => r != null && Odds.IsZombie(r);

        static void Chip(Label chip, PolicyRunner r)
        {
            bool z = Zombie(r);
            chip.text = z ? "Z" : "M";
            chip.EnableInClassList("bh-chip-zombie", z);
            chip.EnableInClassList("bh-chip-matt", !z);
        }

        void LateUpdate()
        {
            if (_frame == null) return;
            _anchors.Tick();
            FitToScreen();
            Refresh();
        }

        /// <summary>One HUD update from the board (LateUpdate; tests call it directly).</summary>
        public void Refresh()
        {
            var b = B;
            if (b == null || _frame == null) return;
            if (b.Heat != _heatSeen) NewHeat(b);
            var rows = b.Rows.ToList();
            bool stageDone = Gauntlet.Active && Gauntlet.CurrentHeatPlayed && _frozen != null;   // wait for Next event
            if (stageDone) rows = _frozen;
            else if (b.BoardState == BoardPhase.Ready) RefreshOdds(rows);
            HandleBetting(b, rows);
            _pbp.Update(b, Time.unscaledTime);

            bool result = stageDone || b.BoardState == BoardPhase.Result;
            _anchors.Sub.text = $"{subtitle.Replace("Event ", "E")} · {b.SubtitleExtra}";
            _clock.text = b.ClockLine;
            _info.text = b.InfoLine;
            // winner banner only for a moment: the result now lives in the standings, the game stays visible
            _banner.text = _slipOpen || (result && Time.unscaledTime - _resultSince > winnerBannerSeconds) ? "" : b.Banner;
            _ticker.text = string.Join("\n", _pbp.Lines.AsEnumerable().Reverse().Take(2));
            _coins.text = offerBets ? $"{Wallet.Coins} coins" + (_bet != null ? $"\nbet {Short(_bet, rows)} @ {Odds.Format(_betOdds)}" : "") : "";

            if (b.BoardState == BoardPhase.Result && _settledHeat != b.Heat) Settle(b, rows);
            _resultBox.style.display = result && _settledHeat == b.Heat ? DisplayStyle.Flex : DisplayStyle.None;
            DrawRows(rows, result);
            UpdateStory(b, rows);

            _primary.style.display = result ? DisplayStyle.Flex : DisplayStyle.None;
            _primary.text = stageDone ? (Gauntlet.IsLast ? "Final standings" : "Next event") : "New heat";
        }

        void NewHeat(IBroadcastBoard b)
        {
            _heatSeen = b.Heat;
            _bet = null;
            _recordHolder = null;
            _heatTopSpeed = 0f;
            _heldFor = 0;
            _oddsBy.Clear();
            _resultBox.Clear();
            if (Gauntlet.Active && Gauntlet.CurrentHeatPlayed) { b.HoldStart = true; return; }   // one heat per gauntlet stage
            if (offerBets) { b.HoldStart = true; _slipOpen = true; _slip.style.display = DisplayStyle.Flex; _slipBuilt = false; }
        }

        bool _slipBuilt;

        void RefreshOdds(List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            var runners = rows.Select(r => r.runner).Where(r => r != null).ToList();
            if (runners.Count != rows.Count) return;
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
                    Chip(AddLabel(btn, "bh-c-chip", ""), runner);
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
            if (_expanded || rows.Count <= n) return rows;
            var pick = Enumerable.Range(0, n - 1).ToList();
            var hot = tension != null && tension.HotDanger >= 0.45f ? tension.Hot : null;
            int extra = hot == null ? -1 : rows.FindIndex(r => r.runner != null && r.runner == hot && !r.bad);
            if (extra < n - 1) extra = _bet == null ? -1 : rows.FindIndex(r => r.runner != null && r.runner == _bet);
            if (extra < n - 1) extra = n - 1;
            pick.Add(extra);
            ranks = pick;
            return pick.Select(i => rows[i]).ToList();
        }

        void DrawRows(List<(int place, string name, string result, bool bad, PolicyRunner runner)> all, bool result)
        {
            var rows = Shown(all, out var ranks);
            while (_rows.Count < rows.Count)
            {
                var row = Add(_rowsBox, "bh-row");
                var place = AddLabel(row, "bh-c-place", "");
                var chip = AddLabel(row, "bh-c-chip", "");
                var name = AddLabel(row, "bh-c-name", "");
                var res = AddLabel(row, "bh-c-result", "");
                var badge = AddLabel(row, "bh-c-badge", "");
                var confBox = Add(row, "bh-c-conf");
                var spark = new Sparkline { capacity = 50, lineWidth = 2f };
                confBox.Add(spark);
                var conf = AddLabel(confBox, "bh-conf-pct", "");
                _rows.Add((row, place, chip, name, badge, res, spark, conf, AddLabel(row, "bh-c-odds", "")));
            }
            // the last column: odds with betting on, else the expand hint on the last row ("8 ▾" / "▴")
            string hint = all.Count <= Mathf.Max(1, collapsedRows) ? "" : _expanded ? "▲" : $"{all.Count} ▼";
            string[] medal = { "gold", "silver", "bronze" };
            for (int i = 0; i < _rows.Count; i++)
            {
                var ui = _rows[i];
                bool on = i < rows.Count;
                ui.row.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
                if (!on) continue;
                var r = rows[i];
                ui.place.text = (r.place > 0 ? r.place : ranks[i] + 1).ToString();
                Chip(ui.chip, r.runner);
                ui.name.text = r.name;
                ui.result.text = r.result;
                bool wr = result && r.runner != null && r.runner == _recordHolder;
                bool bet = r.runner != null && r.runner == _bet;
                ui.badge.text = wr ? "WR" : bet ? "BET" : "";
                ui.badge.EnableInClassList("bh-badge-wr", wr);
                ui.badge.style.display = wr || bet ? DisplayStyle.Flex : DisplayStyle.None;
                ui.odds.text = offerBets ? (r.runner != null && _oddsBy.TryGetValue(r.runner, out var o) ? Odds.Format(o) : "")
                             : i == rows.Count - 1 ? hint : "";
                ui.odds.EnableInClassList("bh-expand", !offerBets);
                var tel = Telemetry(r.runner);
                // confidence only when it tells something: a brain under 90 % (a column of "100%" is noise)
                bool hasConf = tel != null && !float.IsNaN(tel.Confidence) && !r.bad && tel.Confidence < 0.9f;
                ui.conf.text = hasConf ? $"{tel.Confidence * 100f:0}%" : r.bad ? "—" : "";
                ui.conf.style.color = hasConf ? Sparkline.ColorOf(tel.Confidence) : new Color(0.55f, 0.6f, 0.69f);
                if (tel != null) ui.spark.SetValues(tel.ConfidenceHistory);
                ui.spark.style.visibility = hasConf ? Visibility.Visible : Visibility.Hidden;
                ui.row.EnableInClassList("bh-row-hot", tension != null && tension.Hot == r.runner && tension.HotDanger >= 0.45f && !r.bad);
                ui.row.EnableInClassList("bh-row-bad", r.bad);
                ui.row.EnableInClassList("bh-row-first", r.place == 1 && !result);
                ui.row.EnableInClassList("bh-row-bet", bet);
                for (int m = 0; m < 3; m++) ui.row.EnableInClassList("bh-row-" + medal[m], result && r.place == m + 1);
            }
        }

        /// <summary>The result, in the standings (no card over the game): WR line, heat bests, bet, gauntlet points.</summary>
        void Settle(IBroadcastBoard b, List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            _settledHeat = b.Heat;
            _resultSince = Time.unscaledTime;
            _resultBox.Clear();
            var winner = rows.FirstOrDefault(r => r.place == 1);
            if (eventNumber > 0) WorldRecords(b, winner);
            HeatBests(rows);
            if (_bet != null)
            {
                bool won = rows.Any(r => r.runner == _bet && r.place == 1);
                int pay = won ? Mathf.RoundToInt(Wallet.Stake * _betOdds) : 0;
                if (won) Wallet.Coins += pay;
                AddLabel(_resultBox, won ? "bh-card-good" : "bh-card-line",
                    won ? $"Bet on {Short(_bet, rows)} @ {Odds.Format(_betOdds)} pays {pay} coins!" : $"Bet on {Short(_bet, rows)} lost ({Wallet.Stake} coins)");
                _pbp.Add(won ? $"Bet won: +{pay} coins" : "Bet lost");
            }
            if (winner.name != null) _pbp.Add($"{winner.name} wins — {winner.result}");
            if (winner.name != null && screens != null) screens.Banner($"{winner.name} WINS", 4f);
            if (Gauntlet.Active && !Gauntlet.CurrentHeatPlayed)
            {
                Gauntlet.Award(rows.Select(r => (r.name, r.place)));
                _frozen = rows;
                AddLabel(_resultBox, "bh-card-line", $"GAUNTLET {Gauntlet.Index + 1}/{Gauntlet.Events.Length} · " +
                    string.Join(" · ", Gauntlet.Table().Take(4).Select(t => $"{t.lane} {t.points}")));
            }
        }

        AthleteTelemetry Telemetry(PolicyRunner r) =>
            r == null ? null : tension != null ? tension.TelemetryOf(r) : r.GetComponent<AthleteTelemetry>();

        static string Body(PolicyRunner r) => r == null ? "" : Odds.IsZombie(r) ? "ZOMBIE" : "MATT";

        /// <summary>One line: this heat's winning mark against the world record (new record → WR badge + pulse).</summary>
        void WorldRecords(IBroadcastBoard b, (int place, string name, string result, bool bad, PolicyRunner runner) winner)
        {
            var line = Add(_resultBox, "bh-wr");
            AddLabel(line, "bh-wr-title", "WR");
            int rank = 0;
            bool has = b.TryWinningMark(out var v, out var lower, out var text);
            if (has) rank = Records.Submit(eventNumber, v, lower, text, winner.name ?? "", Body(winner.runner), b.Heat + 1);
            var top = Records.Top(eventNumber);
            string holder = top.Count == 0 ? "" : top[0].holder + (string.IsNullOrEmpty(top[0].body) ? "" : $" · {top[0].body}");
            AddLabel(line, "bh-wr-mark", top.Count == 0 ? "no mark yet" : top[0].mark);
            AddLabel(line, "bh-wr-who", holder);
            var tag = AddLabel(line, "bh-wr-tag", rank == 1 ? "NEW RECORD!" : rank > 1 ? $"#{rank} ALL-TIME" : "");
            if (rank == 1)
            {
                _recordHolder = winner.runner;
                _pbp.Add($"WORLD RECORD! {winner.name} — {text}");
                if (arenaAudio != null) arenaAudio.NewRecord();
                if (screens != null) screens.Banner("WORLD RECORD", 7f);
                Tween.Custom(1f, 1.08f, 0.35f, s => tag.style.scale = new Scale(new Vector3(s, s, 1f)), ease: Ease.InOutSine,
                             cycles: 6, cycleMode: CycleMode.Yoyo, useUnscaledTime: true);
            }
            else if (rank > 1) _pbp.Add($"{winner.name} goes #{rank} on the all-time list");
            Tween.Custom(0f, 1f, 0.45f, a => line.style.opacity = a, ease: Ease.OutQuad, startDelay: 0.25f, useUnscaledTime: true);
        }

        /// <summary>The heat's physical bests from telemetry (fastest, most powerful, closest call) as compact badges.</summary>
        void HeatBests(List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            var tel = rows.Select(r => (r.name, r.bad, t: Telemetry(r.runner))).Where(x => x.t != null && x.t.Ready).ToList();
            if (tel.Count == 0) return;
            var box = Add(_resultBox, "bh-bests");
            void Best(string caption, string value) { var c = Add(box, "bh-best"); AddLabel(c, "bh-best-cap", caption); AddLabel(c, "bh-best-val", value); }
            var fast = tel.OrderByDescending(x => x.t.PeakSpeedMps).First();
            bool race = tension == null || tension.kind == BroadcastDirector.Kind.Race;       // stationary events: a speed is a fall
            if (race) Best("TOP", $"{fast.t.PeakSpeedMps:0.0} m/s {fast.name}");
            var strong = tel.OrderByDescending(x => x.t.PeakPowerW).First();
            Best("PWR", $"{strong.t.PeakPowerW:N0} W {strong.name}");
            // closest call = the lowest confidence of an athlete who stayed in (a faller's 0 % is not a "call")
            var close = tel.Where(x => !x.bad && !float.IsNaN(x.t.Confidence)).OrderBy(x => x.t.MinConfidence).FirstOrDefault();
            if (close.t != null && close.t.MinConfidence < 0.5f) Best("CLOSE", $"{close.name} {close.t.MinConfidence * 100f:0}%");
        }

        /// <summary>Story strip (the old stats card and its tiles, merged into one row): chip · name · tag · speed ·
        /// power · confidence with its sparkline.</summary>
        void BuildStory()
        {
            _story = new VisualElement();
            _story.AddToClassList("bh-story");
            _dockBottom.Insert(0, _story);
            _story.pickingMode = PickingMode.Ignore;
            _storyChip = AddLabel(_story, "bh-c-chip", "");
            _storyName = AddLabel(_story, "bh-story-name", "");
            _storyTag = AddLabel(_story, "bh-story-tag", "");
            Add(_story, "bh-story-gap");
            _sSpeed = AddLabel(_story, "bh-story-val", "—");
            _sPower = AddLabel(_story, "bh-story-val", "—");
            var conf = Add(_story, "bh-story-conf");
            _sConf = AddLabel(conf, "bh-story-val", "—");
            _storySpark = new Sparkline { capacity = 60 };
            _storySpark.AddToClassList("bh-story-spark");
            conf.Add(_storySpark);
            _story.style.visibility = Visibility.Hidden;
        }

        void UpdateStory(IBroadcastBoard b, List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            if (_story == null) return;
            if (_director == null) _director = FindAnyObjectByType<BroadcastDirector>();
            // the strip follows the camera's story (the athlete on screen) and refreshes once a second (no flicker),
            // and at once when the phase changes (the winner shows the moment the heat ends)
            if (Time.unscaledTime >= _storyNext || _slipOpen || b.BoardState != _storyPhase)
            {
                _storyNext = Time.unscaledTime + 1f;
                _storyPhase = b.BoardState;
                PolicyRunner who;
                string tag;
                if (b.BoardState == BoardPhase.Result) { who = rows.FirstOrDefault(r => r.place == 1).runner; tag = "WINNER"; }
                else
                {
                    who = _director != null && _director.Subject != null ? _director.Subject
                        : tension != null && tension.Leader != null ? tension.Leader : rows.FirstOrDefault(r => !r.bad).runner;
                    tag = _director != null && _director.Current == BroadcastDirector.Shot.Hot ? "IN TROUBLE"
                        : b.BoardState == BoardPhase.Live ? "LEADER" : "READY";
                }
                var t = Telemetry(who);
                bool show = t != null && t.Ready && !_slipOpen && rows.Any(r => r.runner == who);
                _story.style.visibility = show ? Visibility.Visible : Visibility.Hidden;
                if (!show) _storyFor = null;
                else
                {
                    if (who != _storyFor)
                    {
                        _storyFor = who;
                        Tween.Custom(0.4f, 1f, 0.3f, a => _story.style.opacity = a, useUnscaledTime: true);
                    }
                    var row = rows.First(r => r.runner == who);
                    Chip(_storyChip, who);
                    _storyName.text = row.name;
                    _storyName.tooltip = Body(who);
                    _storyTag.text = tag;
                    _storyTag.EnableInClassList("bh-story-tag-hot", tag == "IN TROUBLE");
                    _sSpeed.text = $"{t.SpeedMps:0.0} m/s";
                    _sPower.text = $"{t.PowerW:N0} W";
                    bool hasConf = !float.IsNaN(t.Confidence);
                    _sConf.text = hasConf ? $"{t.Confidence * 100f:0}%" : "—";
                    _sConf.style.color = hasConf ? Sparkline.ColorOf(t.Confidence) : new Color(0.55f, 0.6f, 0.69f);
                    _storySpark.SetValues(t.ConfidenceHistory);
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

        /// <summary>Scale the 1080-wide layout onto the safe area (HudAnchors.SafeArea, normalised) and give the camera
        /// the gap between the docks as its viewport.</summary>
        public void FitToScreen()
        {
            var panel = _root.panel;
            if (panel == null) return;
            var full = panel.visualTree.layout;
            if (float.IsNaN(full.width) || full.width <= 1 || full.height <= 1) return;
            var safe = HudAnchors.SafeRectIn(full.size);
            float s = safe.width / DesignWidth;
            _safeTop.style.left = 0; _safeTop.style.right = 0; _safeTop.style.top = 0; _safeTop.style.height = Mathf.Max(0f, safe.yMin);
            _safeBottom.style.left = 0; _safeBottom.style.right = 0; _safeBottom.style.top = safe.yMax;
            _safeBottom.style.height = Mathf.Max(0f, full.height - safe.yMax);
            _frame.style.left = safe.xMin;
            _frame.style.top = safe.yMin;
            _frame.style.width = DesignWidth;
            _frame.style.height = safe.height / s;
            _frame.style.transformOrigin = new TransformOrigin(0, 0);
            _frame.style.scale = new Scale(new Vector3(s, s, 1));

            float topH = _dockTop.layout.height, botH = _dockBottom.layout.height;
            if (float.IsNaN(topH) || float.IsNaN(botH) || topH <= 0f) return;      // first frame: not laid out yet
            var n = HudAnchors.SafeArea();
            float y0 = n.yMin + botH * s / full.height, y1 = n.yMax - topH * s / full.height;   // normalised, y up
            if (y1 - y0 < 0.05f) return;
            if (_viewCam == null && Camera.main != null) _viewCam = Camera.main.GetComponent<BroadcastCamera>();
            if (_viewCam != null) _viewCam.SetViewport(new Rect(n.xMin, y0, n.width, y1 - y0));
        }

        void OnDestroy() { if (_viewCam != null) _viewCam.ClearViewport(); }
    }
}
