using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;
using UnityEngine.UIElements;

namespace PoOlympic.Tests
{
    /// <summary>
    /// The whole loop in Play mode, driven through the buttons a player taps:
    ///   menu → PLAY → heat → result → match card (Places / Telemetry) → OK → menu
    ///   menu → another event → PLAY → result → sheet: New heat → second heat → sheet: Main menu → menu
    /// Fails on a dead end (every wait has a limit), a load that is dropped or hangs, and on any error, exception or
    /// warning in the console on the way (errors and exceptions fail a Unity test by themselves; warnings are
    /// collected here).
    /// Run: Test Runner › PlayMode, or  unity command run_tests --mode PlayMode --filter StateFlowTests --async_tests true
    /// </summary>
    public class StateFlowTests
    {
        const float LoadSeconds = 60f, ReadySeconds = 60f, HeatSeconds = 240f;
        readonly List<string> _warnings = new();

        void OnLog(string message, string stack, LogType type)
        {
            if (type == LogType.Warning) _warnings.Add(message);
        }

        [SetUp]
        public void Listen()
        {
            _warnings.Clear();
            Application.logMessageReceived += OnLog;
        }

        [TearDown]
        public void Quiet()
        {
            Application.logMessageReceived -= OnLog;
            GameSettings.SimSpeed = 1f;
        }

        [UnityTest, Timeout(900000)]
        public IEnumerator Menu_Heat_Result_AndBack_Twice()
        {
            // ---- lap 1: the first event, out through the match card
            yield return Menu(first: true);
            var menu = Find<MainMenuController>();
            Assert.Greater(menu.events.Count, 1, "the menu needs two playable events for this loop");
            yield return PlayEvent(0);
            var hud = Find<BroadcastHud>();
            yield return Until(() => hud.Board != null && hud.Board.BoardState == BoardPhase.Result, HeatSeconds, "heat 1 never reached its result");
            yield return Until(() => Button(hud.Frame, "Results") != null, 10f, "no primary action at the result");
            Click(Button(hud.Frame, "Results"));                    // the primary action: skips the medal ceremony
            yield return Until(() => Button(hud.Frame, "OK") != null, 10f, "no match card after Results");
            Click(Button(hud.Frame, "Telemetry"));
            yield return null;
            Click(Button(hud.Frame, "Places"));
            yield return null;
            Click(Button(hud.Frame, "OK"));
            Assert.IsTrue(MeetLineup.Loading, "OK on the match card did not start a load");
            Click(Button(hud.Frame, "OK"));                         // a double tap while loading must be a no-op
            yield return Menu(first: false);

            // ---- lap 2: another event, a second heat from the sheet, out through the sheet
            yield return PlayEvent(Find<MainMenuController>().events.Count - 1);
            hud = Find<BroadcastHud>();
            yield return Until(() => hud.Board.BoardState == BoardPhase.Result, HeatSeconds, "heat 2 never reached its result");
            int heat = hud.Board.Heat;
            yield return Sheet(hud, "New heat");
            yield return Until(() => hud.Board.Heat != heat && hud.Board.BoardState != BoardPhase.Result, ReadySeconds, "New heat did not restart the event");
            yield return Until(() => hud.Board.BoardState == BoardPhase.Live, ReadySeconds, "the second heat never went live");
            hud.Anchors.ShowTab(HudAnchors.SettingsTab);            // the settings page mid-heat, then out
            yield return null;
            hud.Anchors.CloseSheet();
            yield return Sheet(hud, "Main menu");
            yield return Menu(first: false);

            Assert.AreEqual(1f, Time.timeScale, "back in the menu at another time scale");
            Assert.IsFalse(Gauntlet.Active, "a gauntlet is running after single events");
            Assert.IsEmpty(_warnings, "console warnings on the way:\n  " + string.Join("\n  ", _warnings.Distinct().Take(20)));
        }

        // ------------------------------------------------------------------------------------------ steps
        IEnumerator Menu(bool first)
        {
            if (first) Assert.IsTrue(MeetLineup.Load(MeetLineup.MenuScene), "the menu scene is not in Build Settings");
            yield return Until(() => SceneManager.GetActiveScene().name == MeetLineup.MenuScene && !MeetLineup.Loading &&
                                     UnityEngine.Object.FindAnyObjectByType<MainMenuController>() != null, LoadSeconds, "the main menu did not load");
            yield return null;                                      // one frame for the menu to populate and lay out
        }

        IEnumerator PlayEvent(int index)
        {
            var menu = Find<MainMenuController>();
            var root = menu.GetComponent<UIDocument>().rootVisualElement;
            var tiles = root.Query<Button>(className: "event-button").ToList();
            Assert.AreEqual(menu.events.Count, tiles.Count, "event tiles");
            Click(tiles[index]);
            yield return null;
            var play = root.Q<Button>("playButton");
            Assert.IsTrue(play.enabledSelf, $"PLAY is disabled for event {menu.events[index].number:00} ({menu.Anchors.Sub.text})");
            Click(play);
            Assert.IsTrue(MeetLineup.Loading, "PLAY did not start a load");
            string scene = menu.events[index].scene;
            yield return Until(() => SceneManager.GetActiveScene().name == scene && !MeetLineup.Loading &&
                                     UnityEngine.Object.FindAnyObjectByType<BroadcastHud>() != null, LoadSeconds, $"{scene} did not load");
            var hud = Find<BroadcastHud>();
            yield return Until(() => hud.Board != null && hud.Anchors != null, 5f, $"{scene}: the HUD has no board");
            GameSettings.SimSpeed = 2f;                             // the sheet's fastest viewing speed
            yield return Until(() => hud.Board.BoardState == BoardPhase.Live, ReadySeconds, $"{scene}: the heat never went live");
        }

        /// <summary>TR button → the bottom sheet → the action whose label starts with `label`.</summary>
        IEnumerator Sheet(BroadcastHud hud, string label)
        {
            Click(hud.Anchors.MenuButton);
            yield return null;
            Assert.IsTrue(hud.Anchors.SheetOpen, "the menu button did not open the sheet");
            var item = hud.Frame.Query<Button>(className: "ha-sheet-item").ToList().FirstOrDefault(b => b.text.StartsWith(label));
            Assert.IsNotNull(item, $"no '{label}' in the sheet");
            Click(item);
            yield return null;
            Assert.IsFalse(hud.Anchors.SheetOpen, $"the sheet stayed open after '{label}'");
        }

        // ------------------------------------------------------------------------------------------ helpers
        static T Find<T>() where T : UnityEngine.Object
        {
            var o = UnityEngine.Object.FindAnyObjectByType<T>();
            Assert.IsNotNull(o, $"no {typeof(T).Name} in {SceneManager.GetActiveScene().name}");
            return o;
        }

        /// <summary>The shown, enabled button with this text under `root` (null if none).</summary>
        static Button Button(VisualElement root, string text) =>
            root.Query<Button>().ToList().FirstOrDefault(b => b.text == text && b.enabledInHierarchy && HudLayoutAudit.Shown(b));

        /// <summary>A tap as the UI sees it: the button's own submit (the same path as a click, bindings included).</summary>
        static void Click(Button b)
        {
            Assert.IsNotNull(b, "button not on screen");
            using var e = NavigationSubmitEvent.GetPooled();
            e.target = b;
            b.SendEvent(e);
        }

        static IEnumerator Until(Func<bool> done, float seconds, string failure)
        {
            float end = Time.realtimeSinceStartup + seconds;
            while (!done())
            {
                if (Time.realtimeSinceStartup > end) Assert.Fail($"{failure} (waited {seconds:0} s, scene {SceneManager.GetActiveScene().name})");
                yield return null;
            }
        }
    }
}
