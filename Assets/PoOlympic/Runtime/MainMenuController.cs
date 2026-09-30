using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// Main menu (UI Toolkit, Assets/PoOlympic/UI/MainMenu.uxml) — one viewport, nothing scrolls (UI consolidation
    /// 2026-09-29):
    ///   HudAnchors frame  TL POOLYMPICS + the lineup / gauntlet status · TC FPS · TR menu sheet (lineup presets,
    ///                     performance) · BL debug · BR version; bottom centre: the last gauntlet result
    ///   roster            one card per athlete (MeetLineup.Roster): tap = that athlete in all 8 lanes
    ///   lanes             4 × 2 tiles: tap a lane = the next athlete in it
    ///   events            fixed grid of the playable events (filled in by EventScenes.BuildMainMenu from the catalogue)
    ///   detail            the selected event: name, brain, rules and its world record — or, with ★ on, its all-time
    ///                     top 5 (the ★ toggle also puts every event's world record on its tile: the old records board)
    ///   actions           ★ records · GAUNTLET (tapping events builds an ordered series, Gauntlet; one heat per event,
    ///                     points per place) · PLAY (loads the event scene / starts the gauntlet)
    /// Runs in edit mode too, so the full layout shows in the Game view / Device Simulator without entering Play mode.
    /// </summary>
    [ExecuteAlways]
    [RequireComponent(typeof(UIDocument))]
    public class MainMenuController : MonoBehaviour
    {
        [Serializable]
        public class MenuEvent
        {
            public int number;
            public string name;
            public string rules;
            public string brain;
            public string scene;   // scene name in Build Settings
            public string[] lineup; // athlete per lane in the built scene (MATT / ZOMBIE); scenes are built per lineup
        }

        public List<MenuEvent> events = new();
        [Tooltip("Event tiles per row of the grid.")]
        public int columns = 3;
        [Tooltip("Seconds without a tap / key before demo mode starts on its own (0 = never).")]
        public float idleDemoSeconds = 45f;
        float _lastInput;

        readonly string[] _lineup = new string[8];
        readonly List<(Button tile, Label name)> _slots = new();
        readonly List<Button> _eventButtons = new();
        readonly List<Label> _orderBadges = new();
        readonly List<Label> _marks = new();
        readonly List<MenuEvent> _gauntlet = new();
        bool _gauntletMode, _recordsMode;
        Button _gauntletButton, _recordsButton, _play;
        VisualElement _root, _slotsRoot, _detail, _recordTop;
        MenuEvent _selected;
        Label _rules, _eventRecord, _eventTitle, _eventBrain, _last;
        HudAnchors _anchors;

        public HudAnchors Anchors => _anchors;

        void OnEnable()
        {
            _lastInput = Time.unscaledTime;
            Populate();
        }

        void Update()
        {
            if (Application.isPlaying) IdleDemo();
            // the UIDocument rebuilds its tree when its assets reload: repopulate when our elements are gone
            if (_slotsRoot == null || _slotsRoot.panel == null || _slotsRoot.childCount == 0) Populate();
            if (_anchors == null) return;
            _anchors.Tick();
            FitSafeArea();
        }

        /// <summary>Demo mode after idleDemoSeconds without a tap, click, key or gamepad button.</summary>
        void IdleDemo()
        {
            if (DemoRunner.AnyInput(held: true)) _lastInput = Time.unscaledTime;
            else if (idleDemoSeconds > 0f && !DemoMode.Active && Time.unscaledTime - _lastInput > idleDemoSeconds) StartDemo();
        }

        /// <summary>Endless gauntlet of every playable event with random lineups (DemoMode); scenes with a fixed
        /// lineup are left out (their lineup would not match the random one).</summary>
        public void StartDemo()
        {
            if (!Application.isPlaying) return;
            _lastInput = Time.unscaledTime;
            DemoMode.Begin(events.Where(e => e.lineup == null || e.lineup.Length != _lineup.Length).Select(e => (e.number, e.scene)));
        }

        /// <summary>Keep the column inside the safe area (status bar, camera cut-out, navigation bar).</summary>
        public void FitSafeArea()
        {
            var full = _root?.panel?.visualTree.layout ?? Rect.zero;
            if (float.IsNaN(full.width) || full.width <= 1f) return;
            var safe = HudAnchors.SafeRectIn(full.size);
            _root.style.paddingLeft = safe.xMin;
            _root.style.paddingTop = safe.yMin;
            _root.style.paddingRight = Mathf.Max(0f, full.width - safe.xMax);
            _root.style.paddingBottom = Mathf.Max(0f, full.height - safe.yMax);
        }

        public void Populate()
        {
            var doc = GetComponent<UIDocument>();
            var root = doc != null ? doc.rootVisualElement : null;
            var slots = root?.Q<VisualElement>("slots");
            if (slots == null) return;
            _root = root.Q<VisualElement>("root") ?? root;
            var roster = root.Q<VisualElement>("roster");
            var grid = root.Q<VisualElement>("eventList");
            var top = root.Q<VisualElement>("anchorTop");
            var bottom = root.Q<VisualElement>("anchorBottom");
            var overlay = root.Q<VisualElement>("overlay");
            _detail = root.Q<VisualElement>("detail");
            _rules = root.Q<Label>("eventRules");
            _eventTitle = root.Q<Label>("eventTitle");
            _eventBrain = root.Q<Label>("eventBrain");
            _eventRecord = root.Q<Label>("eventRecord");
            _recordTop = root.Q<VisualElement>("recordTop");
            _play = root.Q<Button>("playButton");
            _gauntletButton = root.Q<Button>("gauntletButton");
            _recordsButton = root.Q<Button>("recordsButton");
            _slotsRoot = slots;
            root.style.flexGrow = 1;                     // the document root fills the panel: the column is one viewport tall
            foreach (var v in new[] { slots, roster, grid, top, bottom, overlay }) v.Clear();
            _slots.Clear();
            _eventButtons.Clear();
            _orderBadges.Clear();
            _marks.Clear();

            _anchors = new HudAnchors(top, bottom, overlay, overlay, "POOLYMPICS", "", "v" + Application.version);
            _anchors.SetMenu(MenuItems);
            _last = new Label();
            _last.AddToClassList("menu-last");
            _anchors.BottomCentre.Add(_last);

            foreach (var athlete in MeetLineup.Roster)
            {
                var card = new Button(() => FillAll(athlete)) { tooltip = MeetLineup.Stats(athlete).Replace("\n", " · ") };
                card.AddToClassList("athlete-card");
                card.EnableInClassList("athlete-card--gap", athlete != MeetLineup.Roster[^1]);
                card.Add(Chip(athlete));
                var text = new VisualElement();
                text.AddToClassList("athlete-text");
                text.Add(MakeLabel(athlete, "athlete-name"));
                text.Add(MakeLabel(MeetLineup.Stats(athlete).Split('\n')[0], "athlete-stats"));
                card.Add(text);
                card.Add(MakeLabel("×8", "athlete-fill"));
                roster.Add(card);
            }

            for (int k = 0; k < _lineup.Length; k++)
            {
                string a = MeetLineup.Athletes[k];
                _lineup[k] = Array.IndexOf(MeetLineup.Roster, a) >= 0 ? a : MeetLineup.Roster[0];
                int lane = k;
                var tile = new Button(() => NextAthlete(lane));
                tile.AddToClassList("slot");
                tile.Add(MakeLabel($"LANE {k + 1}", "slot-lane"));
                var name = MakeLabel("", "slot-athlete");
                tile.Add(name);
                slots.Add(tile);
                _slots.Add((tile, name));
            }

            int cols = Mathf.Max(1, columns);
            VisualElement row = null;
            for (int i = 0; i < events.Count || i % cols != 0; i++)
            {
                if (i % cols == 0)
                {
                    row = new VisualElement();
                    row.AddToClassList("event-row");
                    row.EnableInClassList("event-row--gap", i + cols < events.Count);
                    grid.Add(row);
                }
                string gap = i % cols < cols - 1 ? "event-button--gap" : "event-button--end";
                if (i >= events.Count) { var pad = new VisualElement(); pad.AddToClassList("event-button"); pad.AddToClassList(gap); pad.AddToClassList("event-button--empty"); row.Add(pad); continue; }
                var ev = events[i];
                var button = new Button(() => SelectEvent(ev)) { tooltip = ev.brain };
                button.AddToClassList("event-button");
                button.AddToClassList(gap);
                button.Add(MakeLabel($"{ev.number:00}", "event-number"));
                button.Add(MakeLabel(ev.name, "event-name"));
                var mark = MakeLabel("", "event-mark");
                button.Add(mark);
                var order = MakeLabel("", "event-order");
                order.style.display = DisplayStyle.None;
                button.Add(order);
                row.Add(button);
                _eventButtons.Add(button);
                _orderBadges.Add(order);
                _marks.Add(mark);
            }

            _play.clicked -= Play;
            _play.clicked += Play;
            _gauntletButton.clicked -= ToggleGauntlet;
            _gauntletButton.clicked += ToggleGauntlet;
            _recordsButton.clicked -= ToggleRecords;
            _recordsButton.clicked += ToggleRecords;
            RefreshSlots();
            RefreshRecords();
            SelectEvent(events.FirstOrDefault(e => e.number == MeetLineup.EventNumber) ?? events.FirstOrDefault());
        }

        IEnumerable<(string, Action)> MenuItems()
        {
            foreach (var a in MeetLineup.Roster) yield return ($"All {a}", () => FillAll(a));
            if (MeetLineup.Roster.Length > 1)
                yield return ("Alternate " + string.Join(" / ", MeetLineup.Roster), () =>
                {
                    if (LineupFixed) return;
                    for (int k = 0; k < _lineup.Length; k++) _lineup[k] = MeetLineup.Roster[k % MeetLineup.Roster.Length];
                    RefreshSlots();
                });
            yield return (HudAnchors.DebugOn ? "Performance: hide" : "Performance: show", () => HudAnchors.DebugOn = !HudAnchors.DebugOn);
            yield return ($"Demo mode (starts by itself after {idleDemoSeconds:0} s idle)", StartDemo);
            if (DemoSeason.SeriesPlayed > 0) yield return ("Demo season: reset", () => { DemoSeason.Reset(); UpdateStatus(); });
        }

        static Label Chip(string athlete)
        {
            bool z = athlete == "ZOMBIE";
            var chip = MakeLabel(z ? "Z" : athlete.Substring(0, 1), "menu-chip");
            chip.EnableInClassList("menu-chip--zombie", z);
            return chip;
        }

        /// <summary>The selected event's scene fixes the lineup (athlete taps are ignored). Unity serialises a null
        /// array as an empty one, so "any lineup" (roster scenes) arrives here as lineup = [] — only a full 8-lane
        /// lineup is fixed.</summary>
        bool LineupFixed => _selected?.lineup != null && _selected.lineup.Length == _lineup.Length;

        /// <summary>Tap a lane: the next athlete of the roster in it.</summary>
        void NextAthlete(int lane)
        {
            if (LineupFixed) return;                     // lineup fixed by the event scene
            int i = Array.IndexOf(MeetLineup.Roster, _lineup[lane]);
            _lineup[lane] = MeetLineup.Roster[(i + 1) % MeetLineup.Roster.Length];
            RefreshSlots();
        }

        void FillAll(string athlete)
        {
            if (LineupFixed) return;
            for (int k = 0; k < _lineup.Length; k++) _lineup[k] = athlete;
            RefreshSlots();
        }

        void RefreshSlots()
        {
            for (int k = 0; k < _slots.Count; k++)
            {
                var (tile, name) = _slots[k];
                name.text = _lineup[k] ?? "empty";
                tile.EnableInClassList("slot--zombie", _lineup[k] == "ZOMBIE");
                tile.EnableInClassList("slot--fixed", LineupFixed);
            }
            UpdateStatus();
        }

        void ToggleGauntlet()
        {
            _gauntletMode = !_gauntletMode;
            _gauntlet.Clear();
            if (_gauntletMode && _selected != null) _gauntlet.Add(_selected);
            RefreshGauntlet();
        }

        void RefreshGauntlet()
        {
            _gauntletButton.text = _gauntletMode ? $"GAUNTLET {_gauntlet.Count}" : "GAUNTLET";
            _gauntletButton.EnableInClassList("gauntlet-toggle--on", _gauntletMode);
            for (int i = 0; i < _orderBadges.Count && i < events.Count; i++)
            {
                int at = _gauntlet.IndexOf(events[i]);
                _orderBadges[i].style.display = _gauntletMode && at >= 0 ? DisplayStyle.Flex : DisplayStyle.None;
                _orderBadges[i].text = (at + 1).ToString();
                _eventButtons[i].EnableInClassList("event-button--in-gauntlet", _gauntletMode && at >= 0);
            }
            _play.text = _gauntletMode ? $"PLAY {_gauntlet.Count}" : "PLAY";
            UpdateStatus();
        }

        void ToggleRecords()
        {
            _recordsMode = !_recordsMode;
            RefreshRecords();
        }

        /// <summary>★: every tile shows its event's world record and the detail panel the selected event's top 5.</summary>
        void RefreshRecords()
        {
            _recordsButton.EnableInClassList("records-button--on", _recordsMode);
            _detail.EnableInClassList("detail--records", _recordsMode);
            for (int i = 0; i < _marks.Count && i < events.Count; i++)
            {
                var top = Records.Top(events[i].number);
                _marks[i].text = top.Count > 0 ? top[0].mark : "no mark yet";
                _marks[i].EnableInClassList("event-mark--none", top.Count == 0);
                _marks[i].style.display = _recordsMode ? DisplayStyle.Flex : DisplayStyle.None;
            }
            FillTop5();
        }

        void FillTop5()
        {
            _recordTop.Clear();
            if (_selected == null) return;
            var top = Records.Top(_selected.number);
            if (top.Count == 0) _recordTop.Add(MakeLabel("No mark yet: the first valid winning mark sets it.", "record-top-none"));
            for (int i = 0; i < top.Count; i++)
            {
                var row = new VisualElement();
                row.AddToClassList("record-top-row");
                row.Add(MakeLabel((i + 1).ToString(), "record-top-rank"));
                row.Add(MakeLabel(top[i].mark, "record-top-mark"));
                row.Add(MakeLabel(top[i].holder + (string.IsNullOrEmpty(top[i].body) ? "" : $" · {top[i].body}"), "record-top-who"));
                row.Add(MakeLabel(top[i].date ?? "", "record-top-date"));
                _recordTop.Add(row);
            }
        }

        void SelectEvent(MenuEvent ev)
        {
            if (_gauntletMode && ev != null)
            {
                if (!_gauntlet.Remove(ev)) _gauntlet.Add(ev);   // tap = add to the end / remove
                RefreshGauntlet();
            }
            _selected = ev;
            if (LineupFixed) Array.Copy(ev.lineup, _lineup, _lineup.Length);   // the scene's own lineup
            RefreshSlots();
            for (int i = 0; i < _eventButtons.Count; i++)
                _eventButtons[i].EnableInClassList("event-button--selected", events[i] == ev);
            _eventTitle.text = ev != null ? $"{ev.number:00}  {ev.name}" : "";
            _eventBrain.text = ev?.brain ?? "";
            _eventBrain.style.display = string.IsNullOrEmpty(ev?.brain) ? DisplayStyle.None : DisplayStyle.Flex;
            _rules.text = ev?.rules ?? "";
            _eventRecord.text = ev != null && Records.TryGet(ev.number, out _, out var wr) ? $"★ WORLD RECORD  {wr}" : ev != null ? "★ WORLD RECORD  none yet" : "";
            FillTop5();
            UpdateStatus();
        }

        void UpdateStatus()
        {
            if (_play == null || _anchors == null) return;
            bool full = _lineup.All(a => a != null);
            bool loadable = _gauntletMode
                ? _gauntlet.Count > 0 && _gauntlet.All(e => !Application.isPlaying || Application.CanStreamedLevelBeLoaded(e.scene))
                : _selected != null && (!Application.isPlaying || Application.CanStreamedLevelBeLoaded(_selected.scene));
            _play.SetEnabled(full && loadable);
            // the status line lives in the title chip (TL): no separate status label under PLAY
            _anchors.Sub.text = _selected == null ? "No playable events"
                : !full ? "Fill all 8 lanes"
                : !loadable ? $"{_selected.scene} is not in Build Settings"
                : _gauntletMode ? (_gauntlet.Count == 0 ? "Tap events to build the gauntlet"
                    : $"Gauntlet {string.Join(" → ", _gauntlet.Select(e => e.number.ToString("00")))} · 10-8-6-5-4-3-2-1")
                : $"{string.Join(" · ", _lineup.GroupBy(a => a).Select(g => $"{g.Count()}× {g.Key}"))}" + (LineupFixed ? " (event lineup)" : "");
            _last.text = (Gauntlet.LastResult.Length > 0 ? "Last gauntlet: " + Gauntlet.LastResult.Replace("\n", " · ") : "") +
                         (DemoSeason.SeriesPlayed > 0 ? (Gauntlet.LastResult.Length > 0 ? "\n" : "") + "Demo " + DemoSeason.Summary() : "");
        }

        static Label MakeLabel(string text, string cls)
        {
            var l = new Label(text);
            l.AddToClassList(cls);
            return l;
        }

        void Play()
        {
            if (_selected == null || !Application.isPlaying) return;
            if (_gauntletMode && _gauntlet.Count > 0)
            {
                MeetLineup.Set(_lineup, _gauntlet[0].number);
                Gauntlet.Begin(_gauntlet.Select(e => e.number).ToArray(), _gauntlet.Select(e => e.scene).ToArray());
                return;
            }
            Gauntlet.Abandon();
            MeetLineup.Set(_lineup, _selected.number);
            SceneManager.LoadScene(_selected.scene);
        }
    }
}
