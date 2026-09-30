using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic.Tests
{
    /// <summary>
    /// UI consolidation idea 9 — the zero-scroll guard. Lays out the main menu and the broadcast HUD (every phase,
    /// collapsed and expanded standings, the menu sheet open) on off-screen panels at several phone sizes and checks,
    /// with HudLayoutAudit: no ScrollView, nothing outside the safe area, no text under 28 units, the 5 anchors in
    /// their corners, and the camera gap of the HUD large enough to watch the event.
    /// </summary>
    public class PortraitLayoutTests
    {
        const string PanelPath = "Assets/PoOlympic/UI/PoOlympicPanelSettings.asset";
        const string MenuUxml = "Assets/PoOlympic/UI/MainMenu.uxml";
        const string HudUss = "Assets/PoOlympic/UI/BroadcastHud.uss";
        const int TestEvent = 97;                     // records written by the result are cleared in TearDown

        /// <summary>name, pixels, safe area (normalised, y up): a 16:9 phone, the Pixel 9 Pro (status bar + camera
        /// cut-out on top, gesture bar below), a small 16:9 phone, a tall 20:9 phone.</summary>
        public static readonly object[] Phones =
        {
            new object[] { "1080x1920", 1080, 1920, new Rect(0f, 0f, 1f, 1f) },
            new object[] { "Pixel9Pro", 1280, 2856, new Rect(0f, 66f / 2856f, 1f, (2856f - 66f - 140f) / 2856f) },
            new object[] { "720x1280", 720, 1280, new Rect(0f, 0f, 1f, 1f) },
            new object[] { "1080x2400", 1080, 2400, new Rect(0f, 48f / 2400f, 1f, (2400f - 48f - 110f) / 2400f) },
        };

        readonly List<UnityEngine.Object> _trash = new();
        Func<Rect> _safeBefore;

        [SetUp] public void SetUp() { _safeBefore = HudAnchors.SafeArea; Records.Clear(TestEvent); }

        [TearDown]
        public void TearDown()
        {
            HudAnchors.SafeArea = _safeBefore;
            foreach (var o in _trash) if (o != null) UnityEngine.Object.DestroyImmediate(o);
            _trash.Clear();
            Records.Clear(TestEvent);
        }

        // ------------------------------------------------------------------------------------------ harness
        UIDocument MakeDocument(int w, int h, Rect safe, VisualTreeAsset uxml)
        {
            HudAnchors.SafeArea = () => safe;
            var ps = UnityEngine.Object.Instantiate(AssetDatabase.LoadAssetAtPath<PanelSettings>(PanelPath));
            var rt = new RenderTexture(w, h, 0);
            ps.targetTexture = rt;
            var go = new GameObject("PortraitLayoutTest") { hideFlags = HideFlags.HideAndDontSave };
            go.SetActive(false);
            var doc = go.AddComponent<UIDocument>();
            doc.panelSettings = ps;
            doc.visualTreeAsset = uxml;
            _trash.AddRange(new UnityEngine.Object[] { go, ps, rt });
            return doc;
        }

        static void Layout(UIDocument doc)
        {
            var panel = doc.rootVisualElement.panel;
            Assert.NotNull(panel, "document not attached to a panel");
            // edit mode: nothing ticks the runtime panel, so size it from its target texture by hand
            typeof(PanelSettings).GetMethod("ApplyPanelSettings", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                ?.Invoke(doc.panelSettings, null);
            var validate = panel.GetType().GetMethod("ValidateLayout", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            Assert.NotNull(validate, "panel.ValidateLayout not found (UI Toolkit internals changed)");
            validate.Invoke(panel, null);
        }

        static Rect SafeInPanel(UIDocument doc) => HudAnchors.SafeRectIn(doc.rootVisualElement.panel.visualTree.layout.size);

        static void AssertClean(string what, List<string> bad) =>
            Assert.IsEmpty(bad, $"{what}:\n  " + string.Join("\n  ", bad.Take(20)));

        static void AssertAnchors(string what, HudAnchors a, Rect bounds)
        {
            float midX = bounds.center.x, midY = bounds.center.y, tol = bounds.width * 0.12f;
            Rect T(VisualElement e) => e.worldBound;
            Assert.Less(T(a.Title).center.x, midX, $"{what}: title not on the left");
            Assert.Less(T(a.Title).center.y, midY, $"{what}: title not at the top");
            Assert.AreEqual(midX, T(a.Fps).center.x, tol, $"{what}: FPS not centred");
            Assert.Less(T(a.Fps).center.y, midY, $"{what}: FPS not at the top");
            Assert.Greater(T(a.MenuButton).center.x, midX, $"{what}: menu not on the right");
            Assert.Less(T(a.MenuButton).center.y, midY, $"{what}: menu not at the top");
            Assert.Less(T(a.DebugButton).center.x, midX, $"{what}: debug not on the left");
            Assert.Greater(T(a.DebugButton).center.y, midY, $"{what}: debug not at the bottom");
            Assert.Greater(T(a.Version).center.x, midX, $"{what}: version not on the right");
            Assert.Greater(T(a.Version).center.y, midY, $"{what}: version not at the bottom");
            Assert.GreaterOrEqual(a.MenuButton.resolvedStyle.height, 103.5f, $"{what}: menu button under 104");
        }

        // ------------------------------------------------------------------------------------------ main menu
        static readonly string LongRules = "8 runners start on narrow 1m x 1m pedestals. Gusts and falling cubes grow every round. " +
                                           "The last runner to keep base equilibrium without stepping off wins the heat.";

        static List<MainMenuController.MenuEvent> MenuEvents() => new[]
        {
            (1, "The Iron Pedestal"), (5, "The Gust Gauntlet"), (8, "30m All Fours"), (9, "The Inverted Sprint"),
            (10, "Crab Shuffle Relay"), (11, "Slalom Sprint"), (12, "The 360 Turntable"), (13, "Steeplechase Jog"),
            (19, "Terminal Velocity Sprint"), (22, "Emergency Brake"), (23, "The Trench Crawl"),
        }.Select(e => new MainMenuController.MenuEvent
        {
            number = e.Item1 + 70, name = e.Item2, rules = LongRules, brain = "rung2", scene = "Event_Test", lineup = Array.Empty<string>(),
        }).ToList();

        MainMenuController MakeMenu(int w, int h, Rect safe)
        {
            var doc = MakeDocument(w, h, safe, AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(MenuUxml));
            var menu = doc.gameObject.AddComponent<MainMenuController>();
            menu.events = MenuEvents();
            doc.gameObject.SetActive(true);
            Layout(doc);
            menu.Populate();
            Layout(doc);
            menu.FitSafeArea();
            Layout(doc);
            return menu;
        }

        static void Call(object o, string method) =>
            o.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public).Invoke(o, null);

        [TestCaseSource(nameof(Phones))]
        public void Menu_OneViewport(string phone, int w, int h, Rect safe)
        {
            var menu = MakeMenu(w, h, safe);
            var doc = menu.GetComponent<UIDocument>();
            var root = doc.rootVisualElement.Q<VisualElement>("root");
            var bounds = SafeInPanel(doc);
            // the backdrop (menu-root) and the overlay layer fill the whole screen on purpose (the safe-area strips take
            // the menu colour); the content column and whatever the overlay shows must stay inside the safe area
            List<string> Audit() => HudLayoutAudit.Check(root.Q("panel"), bounds, 28f)
                .Concat(root.Q("overlay").Children().SelectMany(c => HudLayoutAudit.Check(c, bounds, 28f))).ToList();
            AssertClean($"{phone} menu", Audit());
            AssertAnchors($"{phone} menu", menu.Anchors, bounds);

            // every event tile fully visible and tappable (≥ 104 tall)
            var tiles = root.Query<Button>(className: "event-button").ToList();
            Assert.AreEqual(11, tiles.Count);
            foreach (var t in tiles) Assert.GreaterOrEqual(t.layout.height, 104f, $"{phone}: event tile under 104");

            Call(menu, "ToggleRecords");                            // ★: marks on the tiles + top 5 in the detail
            Call(menu, "ToggleGauntlet");
            Layout(doc);
            AssertClean($"{phone} menu (records + gauntlet)", Audit());

            Call(menu.Anchors, "ToggleSheet");                      // the TR sheet
            Layout(doc);
            Assert.IsTrue(menu.Anchors.SheetOpen);
            AssertClean($"{phone} menu (sheet open)", Audit());
        }

        // ------------------------------------------------------------------------------------------ broadcast HUD
        sealed class FakeBoard : IBroadcastBoard
        {
            public BoardPhase BoardState { get; set; }
            public int Heat { get; set; }
            public bool HoldStart { get; set; }
            public string SubtitleExtra => "84.39 m · peak 1 s speed";
            public string ClockLine => "22.86 s";
            public string InfoLine => "leader 86.8 / 84.4 m";
            public string Banner => BoardState == BoardPhase.Ready ? "3" : BoardState == BoardPhase.Result ? "L5 WINS" : "";
            public IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows =>
                Enumerable.Range(0, 8).Select(i => (BoardState == BoardPhase.Result ? i + 1 : 0, $"L{i + 1}",
                    i == 7 ? "FELL 12.3 s" : $"{86.8 - i * 0.4:0.0} m · {4.05 - i * 0.01:0.00} m/s", i == 7, (PolicyRunner)null));
            public bool TryWinningMark(out double value, out bool lowerIsBetter, out string text)
            { value = 4.05; lowerIsBetter = false; text = "4.05 m/s"; return BoardState == BoardPhase.Result; }
            public void Restart() { }
        }

        (BroadcastHud hud, UIDocument doc, FakeBoard board) MakeHud(int w, int h, Rect safe)
        {
            var doc = MakeDocument(w, h, safe, null);
            var hud = doc.gameObject.AddComponent<BroadcastHud>();       // not ExecuteAlways: built by hand below
            hud.style = AssetDatabase.LoadAssetAtPath<StyleSheet>(HudUss);
            hud.title = "TERMINAL VELOCITY";
            hud.subtitle = "Event 19";
            hud.eventNumber = TestEvent;
            hud.version = "v0 · test";
            var board = new FakeBoard();
            hud.BoardOverride = board;
            doc.gameObject.SetActive(true);
            hud.Build();
            Layout(doc);
            return (hud, doc, board);
        }

        static void Settle(BroadcastHud hud, UIDocument doc)
        {
            for (int i = 0; i < 3; i++) { hud.Refresh(); Layout(doc); hud.FitToScreen(); Layout(doc); }
        }

        [TestCaseSource(nameof(Phones))]
        public void Hud_OneViewport_EveryPhase(string phone, int w, int h, Rect safe)
        {
            var (hud, doc, board) = MakeHud(w, h, safe);
            var bounds = SafeInPanel(doc);
            float Gap() => hud.Hole.worldBound.height / hud.Frame.worldBound.height;

            foreach (var (phase, expanded, minGap) in new[]
                     {
                         (BoardPhase.Ready, false, 0.6f), (BoardPhase.Live, false, 0.6f), (BoardPhase.Result, false, 0.55f),
                         (BoardPhase.Live, true, 0.5f), (BoardPhase.Result, true, 0.45f),
                     })
            {
                board.BoardState = phase;
                hud.Expanded = expanded;
                Settle(hud, doc);
                string what = $"{phone} HUD {phase}{(expanded ? " (all 8)" : "")}";
                AssertClean(what, HudLayoutAudit.Check(hud.Frame, bounds, 28f));
                AssertAnchors(what, hud.Anchors, bounds);
                Assert.GreaterOrEqual(Gap(), minGap, $"{what}: camera gap {Gap():P0} < {minGap:P0}");
                Assert.AreEqual(bounds.width, hud.Frame.worldBound.width, 1f, $"{what}: frame not fitted to the safe width");
                Assert.AreEqual(bounds.yMin, hud.Frame.worldBound.yMin, 1f, $"{what}: frame not at the safe top");
                Assert.AreEqual(bounds.yMax, hud.Frame.worldBound.yMax, 1f, $"{what}: frame not at the safe bottom");
            }
            // the result is shown in place: no overlay card, the primary action in the bottom bar
            Assert.IsNull(hud.Frame.Q(className: "bh-card"), "result card overlay is back");
            var primary = hud.Anchors.BottomCentre.Q<Button>(className: "bh-btn-primary");
            Assert.IsTrue(HudLayoutAudit.Shown(primary), "no primary action at the result");
            Assert.GreaterOrEqual(primary.resolvedStyle.height, 103.5f);

            Call(hud.Anchors, "ToggleSheet");
            Settle(hud, doc);
            AssertClean($"{phone} HUD sheet open", HudLayoutAudit.Check(hud.Frame, bounds, 28f));
        }
    }
}
