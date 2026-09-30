using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// The fixed 5-slot frame shared by every portrait screen (UI consolidation idea 1, 2026-09-29):
    ///   TL title chip (title + one-line subtitle) · TC FPS pill · TR menu button (drops a sheet of actions; `Extra` is a
    ///   slot left of it, e.g. the race clock) · BL debug button (performance readout, growing up from the corner) ·
    ///   BR version. `BottomCentre` is free for the owner (ticker, primary action, last result).
    /// The debug readout hosts PerfOverlay's line (the overlay hides its own label while a frame is attached) and shares
    /// its on/off state (F3, three-finger tap, the BL button); without a PerfOverlay it shows fps + screen / safe area.
    /// Styles: UI/HudAnchors.uss (imported by BroadcastHud.uss and MainMenu.uss).
    /// </summary>
    public sealed class HudAnchors
    {
        public readonly VisualElement TopBar, BottomBar, Extra, BottomCentre;
        public readonly Label Title, Sub, Fps, Version, DebugPanel;
        public readonly Button MenuButton, DebugButton;
        readonly VisualElement _sheet, _overlay;
        Func<IEnumerable<(string label, Action act)>> _items = () => Array.Empty<(string, Action)>();
        float _acc, _frames, _debugNext;
        float _fps = float.NaN;

        static readonly List<HudAnchors> Live = new();
        const string DebugPref = "poolympic.perfOverlay";

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
        /// <param name="overlay">full-screen parent for the menu sheet</param>
        /// <param name="debugHost">parent of the debug readout (bottom-left of it)</param>
        public HudAnchors(VisualElement top, VisualElement bottom, VisualElement overlay, VisualElement debugHost,
                          string title, string sub, string version)
        {
            _overlay = overlay;
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
            DebugButton = new Button(() => { DebugOn = !DebugOn; _debugNext = 0f; }) { tooltip = "Performance" };
            DebugButton.AddToClassList("ha-icon-btn");
            DebugButton.AddToClassList("ha-debug-btn");
            foreach (var h in new[] { 16f, 28f, 40f })
                Add(DebugButton, "ha-dbg-bar").style.height = h;
            BottomBar.Add(DebugButton);
            BottomCentre = Add(BottomBar, "ha-bottom-centre");
            Version = AddLabel(BottomBar, "ha-version", version);

            _sheet = Add(overlay, "ha-sheet");
            DebugPanel = AddLabel(debugHost, "ha-debug", "");
            DebugPanel.pickingMode = PickingMode.Ignore;
            Live.Add(this);
        }

        /// <summary>The sheet's actions, evaluated each time it opens (labels can show state, e.g. "Show all 8").</summary>
        public void SetMenu(Func<IEnumerable<(string label, Action act)>> items) => _items = items ?? _items;

        public bool SheetOpen => _sheet.ClassListContains("ha-sheet--open");

        public void CloseSheet()
        {
            _sheet.RemoveFromClassList("ha-sheet--open");
            MenuButton.RemoveFromClassList("ha-icon-btn--on");
        }

        void ToggleSheet()
        {
            if (SheetOpen) { CloseSheet(); return; }
            _sheet.Clear();
            foreach (var (label, act) in _items())
            {
                var a = act;
                var b = new Button(() => { CloseSheet(); a?.Invoke(); }) { text = label };
                b.AddToClassList("ha-sheet-item");
                _sheet.Add(b);
            }
            if (_sheet.childCount == 0) return;
            PlaceSheet();
            _sheet.AddToClassList("ha-sheet--open");
            MenuButton.AddToClassList("ha-icon-btn--on");
        }

        void PlaceSheet()
        {
            var r = MenuButton.layout;
            if (float.IsNaN(r.height)) return;
            var p = MenuButton.parent.ChangeCoordinatesTo(_overlay, new Vector2(r.xMin, r.yMax));
            _sheet.style.top = p.y + 8f;
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
            if (!on || Time.unscaledTime < _debugNext) return;
            _debugNext = Time.unscaledTime + 1f;
            var n = SafeArea();
            string screen = $"{Screen.width}×{Screen.height} · safe {n.width * 100f:0}×{n.height * 100f:0}%";
            DebugPanel.text = PerfOverlay.Instance != null && !string.IsNullOrEmpty(PerfOverlay.Line)
                ? PerfOverlay.Line + " · " + screen
                : $"{(float.IsNaN(_fps) ? "—" : _fps.ToString("0"))} fps · {screen}";
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
        /// <summary>Violations under `root` (laid out): ScrollViews, visible elements outside `bounds` (panel space),
        /// and text below `minText` in the element's own layout units (the HUD's 1080-wide design units, the menu's
        /// 1920-tall panel units).</summary>
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
                if (w.xMin < bounds.xMin - slack || w.yMin < bounds.yMin - slack || w.xMax > bounds.xMax + slack || w.yMax > bounds.yMax + slack)
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
