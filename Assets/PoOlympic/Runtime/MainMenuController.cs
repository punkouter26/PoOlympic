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
    /// (filled in by EventScenes.BuildMainMenu from the catalogue); PLAY loads the event scene. Runs in edit mode too,
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
        }

        public List<MenuEvent> events = new();

        readonly string[] _lineup = new string[8];
        readonly List<(Button tile, Label name)> _slots = new();
        readonly List<Button> _eventButtons = new();
        VisualElement _slotsRoot;
        MenuEvent _selected;
        int _lane;
        string _lastAthlete;
        Label _rules, _status;
        Button _play, _fillAll;

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
            _slotsRoot = slots;
            slots.Clear();
            roster.Clear();
            list.Clear();
            _slots.Clear();
            _eventButtons.Clear();
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
                button.Add(number);
                button.Add(title);
                button.Add(brain);
                list.Add(button);
                _eventButtons.Add(button);
            }

            _play.clicked -= Play;
            _play.clicked += Play;
            _fillAll.clicked -= FillAll;
            _fillAll.clicked += FillAll;
            RefreshSlots();
            SelectEvent(events.FirstOrDefault(e => e.number == MeetLineup.EventNumber) ?? events.FirstOrDefault());
        }

        void SelectLane(int lane)
        {
            _lane = lane;
            RefreshSlots();
        }

        /// <summary>Put `athlete` in the selected lane and move the selection on to the next lane.</summary>
        void Assign(string athlete)
        {
            _lastAthlete = athlete;
            _lineup[_lane] = athlete;
            _lane = (_lane + 1) % _lineup.Length;
            RefreshSlots();
        }

        void FillAll()
        {
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

        void SelectEvent(MenuEvent ev)
        {
            _selected = ev;
            for (int i = 0; i < _eventButtons.Count; i++)
                _eventButtons[i].EnableInClassList("event-button--selected", events[i] == ev);
            _rules.text = ev?.rules ?? "";
            UpdateStatus();
        }

        void UpdateStatus()
        {
            if (_play == null) return;
            bool full = _lineup.All(a => a != null);
            bool loadable = _selected != null && (!Application.isPlaying || Application.CanStreamedLevelBeLoaded(_selected.scene));
            _play.SetEnabled(full && loadable);
            _status.text = _selected == null ? "No playable events"
                : !full ? "Fill all 8 lanes"
                : !loadable ? $"{_selected.scene} is not in Build Settings"
                : $"{string.Join(" · ", _lineup.GroupBy(a => a).Select(g => $"{g.Count()}× {g.Key}"))}  ·  event {_selected.number}";
        }

        void Play()
        {
            if (_selected == null || !Application.isPlaying) return;
            MeetLineup.Set(_lineup, _selected.number);
            SceneManager.LoadScene(_selected.scene);
        }
    }
}
