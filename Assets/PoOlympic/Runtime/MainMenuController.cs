using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// Main menu (UI Toolkit, Assets/PoOlympic/UI/MainMenu.uxml): a roster of athlete cards (MeetLineup.Roster), 8 lane
    /// tiles (tap a lane, then an athlete; "Fill all 8" puts the last tapped athlete in every lane), the playable events
    /// (filled in by EventScenes.BuildMainMenu from the catalogue); PLAY loads the event scene. GAUNTLET mode: tapping
    /// events builds an ordered series (Gauntlet), PLAY runs it (one heat per event, points per place). WORLD RECORDS opens
    /// the records board (Records: world record per event, tap for the all-time top 5); the selected event's world
    /// record shows under its rules. Runs in edit mode too,
    /// so the full layout shows in the Game view / Device Simulator without entering Play mode.
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

        readonly string[] _lineup = new string[8];
        readonly List<(Button tile, Label name)> _slots = new();
        readonly List<Button> _eventButtons = new();
        readonly List<Label> _orderBadges = new();
        readonly List<MenuEvent> _gauntlet = new();
        bool _gauntletMode;
        Button _gauntletButton;
        Label _coins;
        VisualElement _slotsRoot;
        MenuEvent _selected;
        int _lane;
        string _lastAthlete;
        Label _rules, _status, _eventRecord;
        Button _play, _fillAll, _recordsButton, _recordsClose;
        VisualElement _recordsPanel;
        ScrollView _recordsList;
        int _recordsOpenEvent = -1;

        void OnEnable() => Populate();

        void Update()
        {
            // the UIDocument rebuilds its tree when its assets reload: repopulate when our elements are gone
            if (_slotsRoot == null || _slotsRoot.panel == null || _slotsRoot.childCount == 0) Populate();
        }

        void Populate()
        {
            var doc = GetComponent<UIDocument>();
            var root = doc != null ? doc.rootVisualElement : null;
            var slots = root?.Q<VisualElement>("slots");
            if (slots == null) return;
            var roster = root.Q<VisualElement>("roster");
            var list = root.Q<ScrollView>("eventList");
            _rules = root.Q<Label>("eventRules");
            _status = root.Q<Label>("status");
            _play = root.Q<Button>("playButton");
            _fillAll = root.Q<Button>("fillAllButton");
            _gauntletButton = root.Q<Button>("gauntletButton");
            _coins = root.Q<Label>("coins");
            _eventRecord = root.Q<Label>("eventRecord");
            _recordsButton = root.Q<Button>("recordsButton");
            _recordsClose = root.Q<Button>("recordsClose");
            _recordsPanel = root.Q<VisualElement>("recordsPanel");
            _recordsList = root.Q<ScrollView>("recordsList");
            _slotsRoot = slots;
            slots.Clear();
            roster.Clear();
            list.Clear();
            _slots.Clear();
            _eventButtons.Clear();
            _orderBadges.Clear();
            _lastAthlete = MeetLineup.Roster[0];
            _lane = 0;

            foreach (var athlete in MeetLineup.Roster)
            {
                var card = new Button(() => Assign(athlete));
                card.AddToClassList("athlete-card");
                var name = new Label(athlete);
                name.AddToClassList("athlete-name");
                var stats = new Label(MeetLineup.Stats(athlete));
                stats.AddToClassList("athlete-stats");
                card.Add(name);
                card.Add(stats);
                roster.Add(card);
            }

            for (int k = 0; k < _lineup.Length; k++)
            {
                string a = MeetLineup.Athletes[k];
                _lineup[k] = Array.IndexOf(MeetLineup.Roster, a) >= 0 ? a : MeetLineup.Roster[0];
                int lane = k;
                var tile = new Button(() => SelectLane(lane));
                tile.AddToClassList("slot");
                var laneLabel = new Label($"LANE {k + 1}");
                laneLabel.AddToClassList("slot-lane");
                var name = new Label();
                name.AddToClassList("slot-athlete");
                tile.Add(laneLabel);
                tile.Add(name);
                slots.Add(tile);
                _slots.Add((tile, name));
            }

            foreach (var ev in events)
            {
                var button = new Button(() => SelectEvent(ev));
                button.AddToClassList("event-button");
                var number = new Label($"{ev.number:00}");
                number.AddToClassList("event-number");
                var title = new Label(ev.name);
                title.AddToClassList("event-name");
                var brain = new Label(ev.brain);
                brain.AddToClassList("event-brain");
                var order = new Label();
                order.AddToClassList("event-order");
                order.style.display = DisplayStyle.None;
                button.Add(number);
                button.Add(title);
                button.Add(brain);
                button.Add(order);
                list.Add(button);
                _eventButtons.Add(button);
                _orderBadges.Add(order);
            }

            _play.clicked -= Play;
            _play.clicked += Play;
            _fillAll.clicked -= FillAll;
            _fillAll.clicked += FillAll;
            if (_gauntletButton != null)
            {
                _gauntletButton.clicked -= ToggleGauntlet;
                _gauntletButton.clicked += ToggleGauntlet;
            }
            if (_recordsButton != null)
            {
                _recordsButton.clicked -= OpenRecords;
                _recordsButton.clicked += OpenRecords;
                _recordsClose.clicked -= CloseRecords;
                _recordsClose.clicked += CloseRecords;
            }
            RefreshSlots();
            SelectEvent(events.FirstOrDefault(e => e.number == MeetLineup.EventNumber) ?? events.FirstOrDefault());
        }

        /// <summary>The selected event's scene fixes the lineup (athlete taps are ignored). Unity serialises a null
        /// array as an empty one, so "any lineup" (roster scenes) arrives here as lineup = [] — only a full 8-lane
        /// lineup is fixed.</summary>
        bool LineupFixed => _selected?.lineup != null && _selected.lineup.Length == _lineup.Length;

        void SelectLane(int lane)
        {
            _lane = lane;
            RefreshSlots();
        }

        /// <summary>Put `athlete` in the selected lane and move the selection on to the next lane.</summary>
        void Assign(string athlete)
        {
            if (LineupFixed) return;                     // lineup fixed by the event scene
            _lastAthlete = athlete;
            _lineup[_lane] = athlete;
            _lane = (_lane + 1) % _lineup.Length;
            RefreshSlots();
        }

        void FillAll()
        {
            if (LineupFixed) return;
            for (int k = 0; k < _lineup.Length; k++) _lineup[k] = _lastAthlete;
            RefreshSlots();
        }

        void RefreshSlots()
        {
            for (int k = 0; k < _slots.Count; k++)
            {
                var (tile, name) = _slots[k];
                name.text = _lineup[k] ?? "empty";
                name.EnableInClassList("slot-athlete--empty", _lineup[k] == null);
                tile.EnableInClassList("slot--selected", k == _lane);
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
            if (_gauntletButton == null) return;
            _gauntletButton.text = _gauntletMode ? $"GAUNTLET: {_gauntlet.Count}" : "GAUNTLET: OFF";
            _gauntletButton.EnableInClassList("gauntlet-toggle--on", _gauntletMode);
            for (int i = 0; i < _orderBadges.Count && i < events.Count; i++)
            {
                int at = _gauntlet.IndexOf(events[i]);
                _orderBadges[i].style.display = _gauntletMode && at >= 0 ? DisplayStyle.Flex : DisplayStyle.None;
                _orderBadges[i].text = (at + 1).ToString();
                _eventButtons[i].EnableInClassList("event-button--in-gauntlet", _gauntletMode && at >= 0);
            }
            if (_play != null) _play.text = _gauntletMode ? $"PLAY GAUNTLET ({_gauntlet.Count})" : "PLAY";
            UpdateStatus();
        }

        void SelectEvent(MenuEvent ev)
        {
            if (_gauntletMode && ev != null)
            {
                if (!_gauntlet.Remove(ev)) _gauntlet.Add(ev);   // tap = add to the end / remove
                RefreshGauntlet();
            }
            _selected = ev;
            if (LineupFixed)   // the scene's own lineup
            {
                Array.Copy(ev.lineup, _lineup, _lineup.Length);
                RefreshSlots();
            }
            for (int i = 0; i < _eventButtons.Count; i++)
                _eventButtons[i].EnableInClassList("event-button--selected", events[i] == ev);
            _rules.text = ev?.rules ?? "";
            if (_eventRecord != null)
                _eventRecord.text = ev != null && Records.TryGet(ev.number, out _, out var wr) ? $"WORLD RECORD  {wr}" : ev != null ? "WORLD RECORD  none yet" : "";
            UpdateStatus();
        }

        void UpdateStatus()
        {
            if (_play == null) return;
            bool full = _lineup.All(a => a != null);
            bool loadable = _gauntletMode
                ? _gauntlet.Count > 0 && _gauntlet.All(e => !Application.isPlaying || Application.CanStreamedLevelBeLoaded(e.scene))
                : _selected != null && (!Application.isPlaying || Application.CanStreamedLevelBeLoaded(_selected.scene));
            _play.SetEnabled(full && loadable);
            _status.text = _selected == null ? "No playable events"
                : !full ? "Fill all 8 lanes"
                : !loadable ? $"{_selected.scene} is not in Build Settings"
                : _gauntletMode ? (_gauntlet.Count == 0 ? "Tap events to build the gauntlet"
                    : $"Gauntlet: {string.Join(" → ", _gauntlet.Select(e => e.number.ToString("00")))}  ·  10-8-6-5-4-3-2-1 points")
                : $"{string.Join(" · ", _lineup.GroupBy(a => a).Select(g => $"{g.Count()}× {g.Key}"))}  ·  event {_selected.number}";
            if (_coins != null)
                _coins.text = Gauntlet.LastResult.Length > 0 ? $"Last gauntlet:\n{Gauntlet.LastResult}" : "";   // no betting: just play
        }

        void OpenRecords()
        {
            _recordsOpenEvent = -1;
            FillRecords();
            _recordsPanel?.AddToClassList("records-panel--open");
        }

        void CloseRecords() => _recordsPanel?.RemoveFromClassList("records-panel--open");

        /// <summary>One card per playable event: world record + holder; the tapped event expands to its top 5.</summary>
        void FillRecords()
        {
            if (_recordsList == null) return;
            _recordsList.Clear();
            foreach (var ev in events)
            {
                var top = Records.Top(ev.number);
                int number = ev.number;
                var card = new Button(() => { _recordsOpenEvent = _recordsOpenEvent == number ? -1 : number; FillRecords(); });
                card.AddToClassList("record-event");
                var head = new VisualElement();
                head.AddToClassList("record-event-head");
                head.Add(MakeLabel($"{ev.number:00}", "record-event-number"));
                head.Add(MakeLabel(ev.name, "record-event-name"));
                var mark = MakeLabel(top.Count > 0 ? top[0].mark : "no mark yet", "record-event-mark");
                mark.EnableInClassList("record-event-mark--none", top.Count == 0);
                head.Add(mark);
                card.Add(head);
                if (top.Count > 0)
                    card.Add(MakeLabel($"{top[0].holder}" + (string.IsNullOrEmpty(top[0].body) ? "" : $" · {top[0].body}") +
                                   (string.IsNullOrEmpty(top[0].date) ? "" : $" · {top[0].date}"), "record-event-holder"));
                if (_recordsOpenEvent == number && top.Count > 0)
                {
                    var list = new VisualElement();
                    list.AddToClassList("record-top");
                    for (int i = 0; i < top.Count; i++)
                    {
                        var row = new VisualElement();
                        row.AddToClassList("record-top-row");
                        row.Add(MakeLabel((i + 1).ToString(), "record-top-rank"));
                        row.Add(MakeLabel(top[i].mark, "record-top-mark"));
                        row.Add(MakeLabel(top[i].holder + (string.IsNullOrEmpty(top[i].body) ? "" : $" · {top[i].body}"), "record-top-who"));
                        row.Add(MakeLabel(top[i].date ?? "", "record-top-date"));
                        list.Add(row);
                    }
                    card.Add(list);
                }
                _recordsList.Add(card);
            }
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
