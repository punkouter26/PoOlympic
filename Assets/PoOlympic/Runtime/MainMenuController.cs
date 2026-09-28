using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// Main menu (UI Toolkit, Assets/PoOlympic/UI/MainMenu.uxml): 8 lane slots cycling through MeetLineup.Roster, the
    /// playable events (filled in by EventScenes.BuildMainMenu from the catalogue), PLAY loads the event scene.
    /// </summary>
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
        readonly List<(Label name, Button prev, Button next)> _slots = new();
        readonly List<Button> _eventButtons = new();
        MenuEvent _selected;
        Label _rules, _status;
        Button _play;

        void OnEnable()
        {
            var root = GetComponent<UIDocument>().rootVisualElement;
            var slots = root.Q<VisualElement>("slots");
            var list = root.Q<ScrollView>("eventList");
            _rules = root.Q<Label>("eventRules");
            _status = root.Q<Label>("status");
            _play = root.Q<Button>("playButton");
            slots.Clear();
            list.Clear();
            _slots.Clear();
            _eventButtons.Clear();

            for (int k = 0; k < _lineup.Length; k++)
            {
                _lineup[k] = Array.IndexOf(MeetLineup.Roster, MeetLineup.Athletes[k]) >= 0 ? MeetLineup.Athletes[k] : MeetLineup.Roster[0];
                int lane = k;
                var row = new VisualElement();
                row.AddToClassList("slot");
                if (k == _lineup.Length - 1) row.AddToClassList("slot--last");
                var laneLabel = new Label($"LANE {k + 1}");
                laneLabel.AddToClassList("slot-lane");
                var prev = new Button(() => Cycle(lane, -1)) { text = "‹" };
                var name = new Label();
                var next = new Button(() => Cycle(lane, +1)) { text = "›" };
                prev.AddToClassList("slot-arrow");
                next.AddToClassList("slot-arrow");
                name.AddToClassList("slot-athlete");
                row.Add(laneLabel);
                row.Add(prev);
                row.Add(name);
                row.Add(next);
                slots.Add(row);
                _slots.Add((name, prev, next));
            }

            foreach (var ev in events)
            {
                var button = new Button(() => Select(ev));
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

            _play.clicked += Play;
            RefreshSlots();
            Select(events.FirstOrDefault(e => e.number == MeetLineup.EventNumber) ?? events.FirstOrDefault());
        }

        void OnDisable()
        {
            if (_play != null) _play.clicked -= Play;
        }

        void Cycle(int lane, int step)
        {
            int i = Array.IndexOf(MeetLineup.Roster, _lineup[lane]);
            _lineup[lane] = MeetLineup.Roster[(i + step + MeetLineup.Roster.Length) % MeetLineup.Roster.Length];
            RefreshSlots();
        }

        void RefreshSlots()
        {
            bool choice = MeetLineup.Roster.Length > 1;
            for (int k = 0; k < _slots.Count; k++)
            {
                var (name, prev, next) = _slots[k];
                name.text = _lineup[k];
                prev.SetEnabled(choice);
                next.SetEnabled(choice);
            }
        }

        void Select(MenuEvent ev)
        {
            _selected = ev;
            for (int i = 0; i < _eventButtons.Count; i++)
                _eventButtons[i].EnableInClassList("event-button--selected", events[i] == ev);
            _rules.text = ev?.rules ?? "";
            bool loadable = ev != null && Application.CanStreamedLevelBeLoaded(ev.scene);
            _play.SetEnabled(loadable);
            _status.text = ev == null ? "No playable events" : loadable ? $"{_lineup.Length} athletes · event {ev.number}" : $"{ev.scene} is not in Build Settings";
        }

        void Play()
        {
            if (_selected == null) return;
            MeetLineup.Set(_lineup, _selected.number);
            SceneManager.LoadScene(_selected.scene);
        }
    }
}
