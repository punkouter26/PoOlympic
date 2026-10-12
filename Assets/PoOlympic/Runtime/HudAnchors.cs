using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// The fixed 5-slot frame shared by every portrait screen (UI consolidation idea 1, 2026-09-29):
    ///   TL title chip (title + one-line subtitle) · TC FPS pill · TR menu button (`Extra` is a slot left of it, e.g.
    ///   the race clock) · BL debug button (performance readout, growing up from the corner) · BR version.
    ///   `BottomCentre` is free for the owner (ticker, primary action, last result).
    /// One modal (2026-10-11): a bottom sheet docked above the bottom bar (thumb reach) with two tabs — Actions (the
    /// owner's items) and Settings (16 chips, one page) — opened by the TR button or a swipe up from the bottom bar,
    /// closed by Close or a tap outside (the scrim also covers the owner's own cards while it is open).
    /// A veil blocks input while MeetLineup loads a scene.
    /// The debug readout hosts PerfOverlay's line (the overlay hides its own label while a frame is attached) and shares
    /// its on/off state (F3, three-finger tap, the BL button); without a PerfOverlay it shows fps + screen / safe area.
    /// Holding the BL button switches the collision-shape view (GameSettings.ColliderView).
    /// Styles: UI/HudAnchors.uss (imported by BroadcastHud.uss and MainMenu.uss).
    /// </summary>
    public sealed class HudAnchors
    {
        public readonly VisualElement TopBar, BottomBar, Extra, BottomCentre;
        public readonly Label Title, Sub, Fps, Version, DebugPanel;
        public readonly Button MenuButton, DebugButton;
        readonly VisualElement _scrim, _sheet, _body, _veil;
        readonly List<Button> _tabs = new();
        int _tab;
        bool _eraseArmed;
        /// <summary>Extra lines for the debug readout (BroadcastHud: the athlete on camera); null / "" = none.</summary>
        public Func<string> DebugExtra;
        Func<IEnumerable<(string label, Action act)>> _items = () => Array.Empty<(string, Action)>();
        float _acc, _frames, _debugNext, _debugDown = -1f;
        float _fps = float.NaN;

        static readonly List<HudAnchors> Live = new();
        const string DebugPref = "poolympic.perfOverlay";
        /// <summary>Full-bleed layers (scrim, loading veil): HudLayoutAudit leaves them out of the safe-area check.</summary>
        public const string BleedClass = "ha-bleed";
        const float HoldSeconds = 0.5f, SwipeUnits = 60f;      // panel units: the panel is 1920 tall on every phone

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => Live.Clear();

        /// <summary>Normalised safe area (0..1, y up). Screen.safeArea by default; tests set their own.</summary>
        public static Func<Rect> SafeArea = () =>
            Screen.width <= 0 || Screen.height <= 0 ? new Rect(0, 0, 1, 1)
            : new Rect(Screen.safeArea.x / Screen.width, Screen.safeArea.y / Screen.height,
                       Screen.safeArea.width / Screen.width, Screen.safeArea.height / Screen.height);

        /// <summary>Is any frame on a live panel (PerfOverlay then leaves the readout to it).</summary>
        public static bool AnyAttached
        {
            get { Live.RemoveAll(a => a.TopBar.panel == null && a.BottomBar.panel == null); return Live.Count > 0; }
        }

        /// <summary>Debug readout on/off; the same switch as PerfOverlay (and its PlayerPrefs key).</summary>
        public static bool DebugOn
        {
            get => PerfOverlay.Instance != null ? PerfOverlay.Instance.Visible : PlayerPrefs.GetInt(DebugPref, 0) == 1;
            set { if (PerfOverlay.Instance != null) PerfOverlay.Instance.Visible = value; else PlayerPrefs.SetInt(DebugPref, value ? 1 : 0); }
        }

        /// <param name="top">the top bar is appended here</param>
        /// <param name="bottom">the bottom bar is appended here</param>
        /// <param name="overlay">full-screen parent for the sheet, its scrim and the loading veil</param>
        /// <param name="debugHost">parent of the debug readout (bottom-left of it)</param>
        public HudAnchors(VisualElement top, VisualElement bottom, VisualElement overlay, VisualElement debugHost,
                          string title, string sub, string version)
        {
            TopBar = Add(top, "ha-bar", "ha-bar-top");
            var left = Add(TopBar, "ha-side");
            var chip = Add(left, "ha-title-chip");
            Title = AddLabel(chip, "ha-title", title);
            Sub = AddLabel(chip, "ha-sub", sub);
            Fps = AddLabel(TopBar, "ha-fps", "— fps");
            Fps.pickingMode = PickingMode.Ignore;
            var right = Add(TopBar, "ha-side", "ha-side-right");
            Extra = Add(right, "ha-extra");
            MenuButton = new Button(ToggleSheet) { tooltip = "Menu" };
            MenuButton.AddToClassList("ha-icon-btn");
            for (int i = 0; i < 3; i++) Add(MenuButton, "ha-burger-bar").pickingMode = PickingMode.Ignore;
            right.Add(MenuButton);

            BottomBar = Add(bottom, "ha-bar", "ha-bar-bottom");
            // tap = the readout, hold = the collision shapes (a key press has no pointer-down: it counts as a tap)
            DebugButton = new Button(() =>
            {
                bool held = _debugDown >= 0f && Time.unscaledTime - _debugDown >= HoldSeconds;
                _debugDown = -1f;
                if (held) GameSettings.ColliderView = !GameSettings.ColliderView;
                else { DebugOn = !DebugOn; _debugNext = 0f; }
            }) { tooltip = "Performance (hold: collision shapes)" };
            DebugButton.RegisterCallback<PointerDownEvent>(_ => _debugDown = Time.unscaledTime, TrickleDown.TrickleDown);
            DebugButton.AddToClassList("ha-icon-btn");
            DebugButton.AddToClassList("ha-debug-btn");
            foreach (var h in new[] { 16f, 28f, 40f })
                Add(DebugButton, "ha-dbg-bar").style.height = h;
            BottomBar.Add(DebugButton);
            BottomCentre = Add(BottomBar, "ha-bottom-centre");
            Version = AddLabel(BottomBar, "ha-version", version);
            SwipeUpOpens(BottomBar);

            _scrim = Add(overlay, "ha-scrim", BleedClass);
            _scrim.RegisterCallback<ClickEvent>(_ => CloseSheet());
            _sheet = Add(overlay, "ha-sheet");
            var head = Add(_sheet, "ha-sheet-head");
            for (int i = 0; i < Tabs.Length; i++)
            {
                int tab = i;
                _tabs.Add(Tab(head, Tabs[i], () => ShowTab(tab)));
            }
            Tab(head, "Close", CloseSheet).AddToClassList("ha-tab-done");
            _body = Add(_sheet, "ha-sheet-body");
            _veil = Add(overlay, "ha-veil", BleedClass);
            AddLabel(_veil, "ha-veil-text", "Loading…");
            DebugPanel = AddLabel(debugHost, "ha-debug", "");
            DebugPanel.pickingMode = PickingMode.Ignore;
            Live.Add(this);
        }

        static Button Tab(VisualElement head, string text, Action act)
        {
            var b = new Button(act) { text = text };
            b.AddToClassList("ha-tab");
            head.Add(b);
            return b;
        }

        /// <summary>A drag up that starts on the bar opens the sheet. The release lands wherever the finger is, so it
        /// is read at the top of the tree (hooked on the first press, when the bar is on its panel).</summary>
        void SwipeUpOpens(VisualElement bar)
        {
            float y0 = float.NaN;
            VisualElement root = null;
            bar.RegisterCallback<PointerDownEvent>(e =>
            {
                y0 = e.position.y;
                if (root != null) return;
                for (root = bar; root.parent != null;) root = root.parent;
                root.RegisterCallback<PointerUpEvent>(u =>
                {
                    if (y0 - u.position.y > SwipeUnits && !SheetOpen) ShowTab(ActionsTab);   // NaN (no press on the bar): false
                    y0 = float.NaN;
                }, TrickleDown.TrickleDown);
            }, TrickleDown.TrickleDown);
        }

        /// <summary>The Actions tab's items, evaluated each time it shows (labels can show state, e.g. "Show all 8").</summary>
        public void SetMenu(Func<IEnumerable<(string label, Action act)>> items) => _items = items ?? _items;

        // ---------------------------------------------------------------- the one sheet: Actions | Settings
        public bool SheetOpen => _sheet.ClassListContains("ha-sheet--open");

        static readonly string[] Tabs = { "Actions", "Settings" };
        public const int ActionsTab = 0, SettingsTab = 1;
        static readonly float[] Levels = { 1f, 0.75f, 0.5f, 0.25f, 0f };
        const int SettingsPerRow = 4;

        static string Pct(float v) => $"{v * 100f:0}%";
        static string OnOff(bool v) => v ? "On" : "Off";
        /// <summary>A level chip steps down 100 → 75 → 50 → 25 → 0 % and wraps (one tap target, no slider to drag).</summary>
        static float NextLevel(float v) => Levels[(Array.FindIndex(Levels, l => l <= v + 0.01f) + 1) % Levels.Length];

        /// <summary>Every option as a chip: short caption, what it is (tooltip), current value, what a tap does.
        /// Rows of four: audio · display · display / game · game.</summary>
        (string cap, string tip, Func<string> value, Action tap)[] Settings() => new (string, string, Func<string>, Action)[]
        {
            ("Master", "Master volume", () => Pct(GameSettings.VolMaster), () => GameSettings.VolMaster = NextLevel(GameSettings.VolMaster)),
            ("Crowd", "Crowd volume", () => Pct(GameSettings.VolCrowd), () => GameSettings.VolCrowd = NextLevel(GameSettings.VolCrowd)),
            ("Impacts", "Impacts and cues", () => Pct(GameSettings.VolSfx), () => GameSettings.VolSfx = NextLevel(GameSettings.VolSfx)),
            ("Voice", "Announcer", () => Pct(GameSettings.VolAnnouncer), () => GameSettings.VolAnnouncer = NextLevel(GameSettings.VolAnnouncer)),

            ("Shake", "Camera shake on impacts", () => Pct(GameSettings.CameraShake), () => GameSettings.CameraShake = NextLevel(GameSettings.CameraShake)),
            ("Zoom", "Zoom in on trouble", () => OnOff(GameSettings.TensionZoom), () => GameSettings.TensionZoom = !GameSettings.TensionZoom),
            ("Blur", "Focus blur on close-ups", () => OnOff(GameSettings.FocusBlur), () => GameSettings.FocusBlur = !GameSettings.FocusBlur),
            ("Marks", "Impact marks", () => OnOff(GameSettings.ImpactMarks), () => GameSettings.ImpactMarks = !GameSettings.ImpactMarks),

            ("Vibrate", "Vibrate on impacts", () => OnOff(GameSettings.Haptics), () => GameSettings.Haptics = !GameSettings.Haptics),
            ("Shapes", "Show collision shapes (or hold the debug button)", () => OnOff(GameSettings.ColliderView), () => GameSettings.ColliderView = !GameSettings.ColliderView),
            ("Bets", "Betting slip before heats", () => OnOff(GameSettings.OfferBets), () => GameSettings.OfferBets = !GameSettings.OfferBets),
            ("Weather", "Real weather (wind, rain)", () => OnOff(GameSettings.RealWeather), () => GameSettings.RealWeather = !GameSettings.RealWeather),

            ("AI talk", "AI commentator", () => AiCommentator.HasKey ? OnOff(GameSettings.AiCommentary) : "No key", () => GameSettings.AiCommentary = !GameSettings.AiCommentary),
            ("Debug", "Debug readout (TAB, or the debug button)", () => OnOff(DebugOn), () => DebugOn = !DebugOn),
            ("Log", "Log every action to file", () => OnOff(GameSettings.LogActions), () => GameSettings.LogActions = !GameSettings.LogActions),
            ("Erase", "Erase coins and records", () => _eraseArmed ? "Tap again" : "Save", () =>
            {
                if (_eraseArmed) GameSettings.ResetSave(Enumerable.Range(1, 40));
                _eraseArmed = !_eraseArmed;
            }),
        };

        /// <summary>Open the sheet on a tab (ActionsTab / SettingsTab), over whatever else the overlay holds.</summary>
        public void ShowTab(int tab)
        {
            _tab = Mathf.Clamp(tab, 0, Tabs.Length - 1);
            _eraseArmed = false;
            Fill();
            PlaceSheet();
            _scrim.BringToFront();
            _sheet.BringToFront();
            _scrim.AddToClassList("ha-scrim--on");
            _sheet.AddToClassList("ha-sheet--open");
            MenuButton.AddToClassList("ha-icon-btn--on");
        }

        public void CloseSheet()
        {
            _scrim.RemoveFromClassList("ha-scrim--on");
            _sheet.RemoveFromClassList("ha-sheet--open");
            MenuButton.RemoveFromClassList("ha-icon-btn--on");
        }

        void ToggleSheet() { if (SheetOpen) CloseSheet(); else ShowTab(ActionsTab); }

        void Fill()
        {
            for (int i = 0; i < _tabs.Count; i++) _tabs[i].EnableInClassList("ha-tab--on", i == _tab);
            _body.Clear();
            if (_tab == ActionsTab)
            {
                foreach (var (label, act) in _items())
                {
                    var a = act;
                    var b = new Button(() => { CloseSheet(); a?.Invoke(); }) { text = label };
                    b.AddToClassList("ha-sheet-item");
                    _body.Add(b);
                }
                return;
            }
            var all = Settings();
            VisualElement row = null;
            for (int i = 0; i < all.Length; i++)
            {
                if (i % SettingsPerRow == 0) row = Add(_body, "ha-set-row");
                var (cap, tip, value, tap) = all[i];
                string v = value();
                var chip = new Button(() => { tap(); Fill(); }) { tooltip = tip };
                chip.AddToClassList("ha-set");
                chip.EnableInClassList("ha-set--off", v is "Off" or "0%" or "No key" or "Save");
                AddLabel(chip, "ha-set-cap", cap).pickingMode = PickingMode.Ignore;
                AddLabel(chip, "ha-set-value", v).pickingMode = PickingMode.Ignore;
                row.Add(chip);
            }
        }

        /// <summary>Docked just above the bottom bar (so inside the safe area wherever the overlay reaches).</summary>
        void PlaceSheet()
        {
            var overlay = _sheet.parent;
            float h = overlay.layout.height;
            if (float.IsNaN(h) || float.IsNaN(BottomBar.layout.height)) return;
            _sheet.style.bottom = Mathf.Max(0f, h - BottomBar.ChangeCoordinatesTo(overlay, Vector2.zero).y) + 8f;
        }

        /// <summary>Per frame: FPS (1 s average, unscaled — hit-stop does not read as a stall) and the debug readout.</summary>
        public void Tick()
        {
            _acc += Time.unscaledDeltaTime;
            _frames++;
            if (_acc >= 0.5f)
            {
                _fps = _frames / _acc;
                _acc = _frames = 0f;
                Fps.text = $"{_fps:0} fps";
                float target = PerfOverlay.TargetFps;
                Fps.EnableInClassList("ha-fps-mid", _fps < target * 0.9f && _fps >= target * 0.5f);
                Fps.EnableInClassList("ha-fps-low", _fps < target * 0.5f);
            }
            if (SheetOpen) PlaceSheet();
            bool loading = MeetLineup.Loading;
            if (loading) _veil.BringToFront();
            _veil.EnableInClassList("ha-veil--on", loading);
            // the debug key on screens without a PerfOverlay (it reads the key itself where there is one)
            if (PerfOverlay.Instance == null && Application.isPlaying && PerfOverlay.DebugKeyPressed) { DebugOn = !DebugOn; _debugNext = 0f; }

            bool on = DebugOn;
            DebugPanel.EnableInClassList("ha-debug--on", on);
            // grow up from BL, above the bottom bar when the host reaches behind it (menu overlay)
            var host = DebugPanel.parent;
            if (on && host != null && !float.IsNaN(host.layout.height))
            {
                float barTop = BottomBar.ChangeCoordinatesTo(host, Vector2.zero).y;
                DebugPanel.style.bottom = Mathf.Max(0f, host.layout.height - barTop) + 16f;
            }
            DebugButton.EnableInClassList("ha-icon-btn--on", on);
            DebugButton.EnableInClassList("ha-debug-btn--shapes", GameSettings.ColliderView);
            if (!on || Time.unscaledTime < _debugNext) return;
            _debugNext = Time.unscaledTime + 1f;
            var n = SafeArea();
            string screen = $"{Screen.width}×{Screen.height} · safe {n.width * 100f:0}×{n.height * 100f:0}%";
            string extra = DebugExtra?.Invoke();
            DebugPanel.text = (PerfOverlay.Instance != null && !string.IsNullOrEmpty(PerfOverlay.Line)
                ? PerfOverlay.Line + " · " + screen
                : $"{(float.IsNaN(_fps) ? "—" : _fps.ToString("0"))} fps · {screen}") + (string.IsNullOrEmpty(extra) ? "" : "\n" + extra);
        }

        /// <summary>GUI-space rectangle (pixels, y down) for the IMGUI dev HUDs (testbeds, the stadium hub): the camera's
        /// view, kept inside the screen's safe area (status bar, camera cut-out, gesture bar).</summary>
        public static Rect GuiArea(Camera cam)
        {
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var n = SafeArea();
            var s = new Rect(n.x * Screen.width, n.y * Screen.height, n.width * Screen.width, n.height * Screen.height);
            float x0 = Mathf.Max(r.xMin, s.xMin), x1 = Mathf.Min(r.xMax, s.xMax), y0 = Mathf.Max(r.yMin, s.yMin), y1 = Mathf.Min(r.yMax, s.yMax);
            return new Rect(x0, Screen.height - y1, Mathf.Max(0f, x1 - x0), Mathf.Max(0f, y1 - y0));
        }

        /// <summary>Panel-space rectangle of the safe area inside a panel of the given size (y down).</summary>
        public static Rect SafeRectIn(Vector2 panelSize)
        {
            var n = SafeArea();
            return new Rect(n.xMin * panelSize.x, (1f - n.yMax) * panelSize.y, n.width * panelSize.x, n.height * panelSize.y);
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
    }

    /// <summary>
    /// Zero-scroll audit (UI consolidation idea 9): everything interactive or readable must sit inside one viewport.
    /// Used by the EditMode layout tests at several phone sizes; also callable at runtime (PoOlympic › UI menu).
    /// </summary>
    public static class HudLayoutAudit
    {
        /// <summary>Violations under `root` (laid out): ScrollViews, visible elements outside `bounds` (panel space;
        /// the full-bleed scrim and loading veil are left out), and text below `minText` in the element's own layout
        /// units (the HUD's 1080-wide design units, the menu's 1920-tall panel units).</summary>
        public static List<string> Check(VisualElement root, Rect bounds, float minText)
        {
            var bad = new List<string>();
            foreach (var sv in root.Query<ScrollView>().ToList())
                if (Shown(sv)) bad.Add($"ScrollView '{sv.name}'");
            foreach (var e in root.Query<VisualElement>().ToList())
            {
                if (!Shown(e)) continue;
                var w = e.worldBound;
                if (w.width < 1f || w.height < 1f) continue;
                const float slack = 1.5f;
                if (!e.ClassListContains(HudAnchors.BleedClass) && (w.xMin < bounds.xMin - slack || w.yMin < bounds.yMin - slack || w.xMax > bounds.xMax + slack || w.yMax > bounds.yMax + slack))
                    bad.Add($"outside: {Describe(e)} {w} (bounds {bounds})");
                if (e is TextElement t && !string.IsNullOrEmpty(t.text) && t.resolvedStyle.fontSize < minText - 0.01f)
                    bad.Add($"text {t.resolvedStyle.fontSize:0} < {minText:0}: {Describe(e)} '{Trim(t.text)}'");
            }
            return bad;
        }

        /// <summary>Visible: displayed, not hidden, and every ancestor too.</summary>
        public static bool Shown(VisualElement e)
        {
            for (var v = e; v != null; v = v.parent)
                if (v.resolvedStyle.display == DisplayStyle.None || v.resolvedStyle.visibility == Visibility.Hidden || v.resolvedStyle.opacity <= 0.01f)
                    return false;
            return true;
        }

        static string Describe(VisualElement e) =>
            (string.IsNullOrEmpty(e.name) ? e.GetType().Name : "#" + e.name) + (e.GetClasses().Any() ? "." + string.Join(".", e.GetClasses()) : "");

        static string Trim(string s) => s.Length > 24 ? s.Substring(0, 24) + "…" : s;
    }
}
