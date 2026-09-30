using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Event HUD for the solo Iron Pedestal (IMGUI dev screen) on the shared anchors (HudAnchors, UI consolidation
    /// 2026-09-29): TL title + timer / telemetry · TC FPS · TR menu + attempt stats · BL debug controls · BR version ·
    /// centre countdown / result banner. Laid out inside the 9:16 camera viewport.
    /// </summary>
    public class EventHud : MonoBehaviour
    {
        public IronPedestalEvent ev;
        public string title = "IRON PEDESTAL";
        public string subtitle = "Event 1 · survive 20 s";
        public string version = "v0 · rung 0 brain";
        float _fps;
        GUIStyle _big, _mid, _small, _panel;

        void Update() => _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(1e-4f, Time.unscaledDeltaTime), 0.05f);

        void Styles(float scale)
        {
            _big = new GUIStyle(GUI.skin.label) { fontSize = Mathf.RoundToInt(64 * scale), fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
            _mid = new GUIStyle(GUI.skin.label) { fontSize = Mathf.RoundToInt(26 * scale), fontStyle = FontStyle.Bold };
            _small = new GUIStyle(GUI.skin.label) { fontSize = Mathf.RoundToInt(16 * scale) };
            _panel = new GUIStyle(GUI.skin.box);
        }

        void OnGUI()
        {
            if (ev == null) return;
            var cam = Camera.main;
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var a = new Rect(r.x, Screen.height - r.yMax, r.width, r.height); // GUI space is y-down
            float s = Mathf.Clamp(a.width / 540f, 0.6f, 2f);
            Styles(s);
            float pad = 12 * s;

            // TL — title
            GUI.Label(new Rect(a.x + pad, a.y + pad, a.width * 0.6f, 34 * s), title, _mid);
            GUI.Label(new Rect(a.x + pad, a.y + pad + 32 * s, a.width * 0.6f, 22 * s), subtitle, _small);

            // TL (under the title) — timer / telemetry
            float remaining = ev.Current == IronPedestalEvent.Phase.Live ? Mathf.Max(0, ev.durationSeconds - ev.LiveTime)
                            : ev.Current == IronPedestalEvent.Phase.Ready ? ev.durationSeconds : Mathf.Max(0, ev.durationSeconds - ev.LiveTime);
            var tl = new Rect(a.x + pad, a.y + pad + 60 * s, 180 * s, 70 * s);
            GUI.Box(tl, GUIContent.none, _panel);
            GUI.Label(new Rect(tl.x, tl.y + 2 * s, tl.width, 40 * s), $"{remaining:0.0}s", new GUIStyle(_mid) { alignment = TextAnchor.MiddleCenter, fontSize = Mathf.RoundToInt(32 * s) });
            GUI.Label(new Rect(tl.x, tl.y + 40 * s, tl.width, 24 * s), $"gusts {ev.Gusts} · cubes {ev.Cubes}", new GUIStyle(_small) { alignment = TextAnchor.MiddleCenter });

            // TC — FPS
            GUI.Label(new Rect(a.x + a.width * 0.5f - 60 * s, a.y + pad, 120 * s, 30 * s), $"{_fps:F0} fps", new GUIStyle(_mid) { alignment = TextAnchor.MiddleCenter });

            // TR — menu + attempt stats
            var tr = new GUIStyle(_small) { alignment = TextAnchor.UpperRight };
            float my = a.y + pad;
            if (MeetLineup.MenuAvailable)
            {
                if (GUI.Button(new Rect(a.xMax - 96 * s - pad, my, 96 * s, 40 * s), "Menu")) MeetLineup.ReturnToMenu();
                my += 46 * s;
            }
            GUI.Label(new Rect(a.xMax - 200 * s - pad, my, 200 * s, 70 * s),
                $"attempt {ev.Attempt + 1}  (seed {ev.seed + ev.Attempt})\nwins {ev.Wins} / {ev.Attempt + (ev.Current == IronPedestalEvent.Phase.Result ? 1 : 0)}\nbest {ev.BestTime:0.0} s", tr);

            // centre banner
            string banner = ev.Current switch
            {
                IronPedestalEvent.Phase.Ready => Mathf.CeilToInt(ev.countdownSeconds - ev.PhaseTime).ToString(),
                IronPedestalEvent.Phase.Result => ev.Outcome == "SURVIVED" ? "SURVIVED!" : $"{ev.Outcome}\n{ev.LiveTime:0.00} s",
                _ => ev.LiveTime < 0.8f ? "GO!" : "",
            };
            if (banner.Length > 0)
                GUI.Label(new Rect(a.x, a.y + a.height * 0.30f, a.width, 160 * s), banner, _big);

            // BL — debug controls
            float bw = 96 * s, bh = 40 * s, by = a.yMax - bh - pad;
            if (GUI.Button(new Rect(a.x + pad, by, bw, bh), "Restart")) ev.Restart();
            if (GUI.Button(new Rect(a.x + pad + (bw + 6 * s), by, bw, bh), "Gust")) ev.cubes.Shove(ev.runner, Random.insideUnitCircle.normalized * ev.gustDv);
            if (GUI.Button(new Rect(a.x + pad + 2 * (bw + 6 * s), by, bw, bh), "Drop cube")) ev.cubes.DropOnAthlete(ev.runner, ev.cubeDropHeight);

            // BR — version
            GUI.Label(new Rect(a.xMax - 180 * s - pad, a.yMax - 26 * s - pad, 180 * s, 26 * s), version, tr);
        }
    }
}
