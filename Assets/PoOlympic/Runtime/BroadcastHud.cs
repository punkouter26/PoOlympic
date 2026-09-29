using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// D3 broadcast HUD (UI Toolkit, 9:16): replaces the IMGUI placeholders for every 8-athlete event.
    ///   top bar      event title · subtitle · clock + info line
    ///   standings    place · lane (body chip) · live result · win odds (from traits, Odds) · your bet
    ///   betting slip each new heat waits in Ready (IBroadcastBoard.HoldStart) until a bet is placed / skipped, or
    ///                betWindowSeconds pass; stake Wallet.Stake coins on one athlete, paid at the decimal odds
    ///   banner       countdown / GO
    ///   ticker       play-by-play lines (Commentary: start, lead changes, athletes out, result)
    ///   result card  podium · bet outcome · record (Records) · gauntlet points (Gauntlet) · New heat / Next / Menu
    /// The layout is authored at 1080 px wide and scaled onto the letterboxed camera rect (BroadcastCamera).
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
        public bool bettingWindow = true;
        public float betWindowSeconds = 12f;

        IBroadcastBoard B => board as IBroadcastBoard;
        Odds.Model _odds;
        readonly Commentary _pbp = new();
        VisualElement _root, _frame, _slip, _card, _rowsBox, _slipRows, _cardBody;
        Label _title, _sub, _clock, _info, _banner, _ticker, _version, _coins, _slipTimer, _cardTitle;
        Button _cardNewHeat;
        List<(int place, string name, string result, bool bad, PolicyRunner runner)> _frozen;   // gauntlet: the scored heat
        readonly List<(VisualElement row, Label place, Label chip, Label name, Label result, Label odds)> _rows = new();
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
            _frame = Add(_root, "bh-frame");
            _frame.pickingMode = PickingMode.Ignore;

            var top = Add(_frame, "bh-top");
            var titles = Add(top, "bh-titles");
            _title = AddLabel(titles, "bh-title", title);
            _sub = AddLabel(titles, "bh-sub", subtitle);
            var clockBox = Add(top, "bh-clockbox");
            _clock = AddLabel(clockBox, "bh-clock", "0.00 s");
            _info = AddLabel(clockBox, "bh-info", "");

            var standings = Add(_frame, "bh-standings");
            var head = Add(standings, "bh-row", "bh-head");
            AddLabel(head, "bh-c-place", "#");
            AddLabel(head, "bh-c-chip", "");
            AddLabel(head, "bh-c-name", "LANE");
            AddLabel(head, "bh-c-result", "RESULT");
            AddLabel(head, "bh-c-odds", "ODDS");
            _rowsBox = Add(standings, "bh-rows");

            _banner = AddLabel(_frame, "bh-banner", "");
            _banner.pickingMode = PickingMode.Ignore;

            var bottom = Add(_frame, "bh-bottom");
            _ticker = AddLabel(bottom, "bh-ticker", "");
            var bar = Add(bottom, "bh-bar");
            Button(bar, "New heat", () => { CloseSlip(); B?.Restart(); }, "bh-btn-small");
            if (MeetLineup.MenuAvailable) Button(bar, "Menu", () => { Gauntlet.Abandon(); MeetLineup.ReturnToMenu(); }, "bh-btn-small");
            _coins = AddLabel(bar, "bh-coins", "");
            _version = AddLabel(bar, "bh-version", version);

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
            var cardBar = Add(_card, "bh-slip-bar");
            _cardNewHeat = Button(cardBar, "New heat", () => { _card.style.display = DisplayStyle.None; B?.Restart(); }, "bh-btn");
            _card.style.display = DisplayStyle.None;
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
            FitToCamera();
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
            _ticker.text = string.Join("\n", _pbp.Lines.AsEnumerable().Reverse().Take(3));
            _coins.text = $"{Wallet.Coins} coins" + (_bet != null ? $" · bet {Short(_bet, rows)} @ {Odds.Format(_betOdds)}" : "");
            DrawRows(rows);

            if (b.BoardState == BoardPhase.Result && _settledHeat != b.Heat) Settle(b, rows);
            if (b.BoardState != BoardPhase.Result && _card.style.display == DisplayStyle.Flex && !Gauntlet.Active) _card.style.display = DisplayStyle.None;
        }

        void NewHeat(IBroadcastBoard b)
        {
            _heatSeen = b.Heat;
            _bet = null;
            _heldFor = 0;
            _oddsBy.Clear();
            if (Gauntlet.Active && Gauntlet.CurrentHeatPlayed) { b.HoldStart = true; return; }   // one heat per gauntlet stage
            if (bettingWindow) { b.HoldStart = true; _slipOpen = true; _slip.style.display = DisplayStyle.Flex; _slipBuilt = false; }
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

        void DrawRows(List<(int place, string name, string result, bool bad, PolicyRunner runner)> rows)
        {
            while (_rows.Count < rows.Count)
            {
                var row = Add(_rowsBox, "bh-row");
                _rows.Add((row, AddLabel(row, "bh-c-place", ""), AddLabel(row, "bh-c-chip", ""), AddLabel(row, "bh-c-name", ""),
                           AddLabel(row, "bh-c-result", ""), AddLabel(row, "bh-c-odds", "")));
            }
            for (int i = 0; i < _rows.Count; i++)
            {
                var ui = _rows[i];
                bool on = i < rows.Count;
                ui.row.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
                if (!on) continue;
                var r = rows[i];
                bool z = Odds.IsZombie(r.runner);
                ui.place.text = (r.place > 0 ? r.place : i + 1).ToString();
                ui.chip.text = z ? "Z" : "M";
                ui.chip.EnableInClassList("bh-chip-zombie", z);
                ui.chip.EnableInClassList("bh-chip-matt", !z);
                ui.name.text = r.name;
                ui.result.text = r.result;
                ui.odds.text = _oddsBy.TryGetValue(r.runner, out var o) ? Odds.Format(o) : "";
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
            // record
            if (eventNumber > 0 && b.TryWinningMark(out var v, out var lower, out var text))
            {
                bool isNew = Records.Submit(eventNumber, v, lower, text, winner.name ?? "");
                if (isNew) { AddLabel(_cardBody, "bh-card-good", $"NEW RECORD  {text}"); _pbp.Add($"New event record: {text}!"); }
                else if (Records.TryGet(eventNumber, out _, out var rec)) AddLabel(_cardBody, "bh-card-line", $"Record: {rec}");
            }
            if (winner.name != null) _pbp.Add($"{winner.name} wins — {winner.result}");
            // gauntlet
            _cardNewHeat.style.display = Gauntlet.Active ? DisplayStyle.None : DisplayStyle.Flex;
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

        /// <summary>Scale the 1080-wide layout onto the camera's (letterboxed) pixel rect.</summary>
        void FitToCamera()
        {
            var panel = _root.panel;
            var cam = Camera.main;
            if (panel == null) return;
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var tl = RuntimePanelUtils.ScreenToPanel(panel, new Vector2(r.xMin, Screen.height - r.yMax));
            var br = RuntimePanelUtils.ScreenToPanel(panel, new Vector2(r.xMax, Screen.height - r.yMin));
            float w = br.x - tl.x, h = br.y - tl.y;
            if (w <= 1 || h <= 1) return;
            float s = w / DesignWidth;
            _frame.style.left = tl.x;
            _frame.style.top = tl.y;
            _frame.style.width = DesignWidth;
            _frame.style.height = h / s;
            _frame.style.transformOrigin = new TransformOrigin(0, 0);
            _frame.style.scale = new Scale(new Vector3(s, s, 1));
        }
    }
}
