using System.Linq;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>360 Turntable HUD (IMGUI placeholder, same layout as RaceHud): title · clock / direction · standings with
    /// turns / time / drift and traits · countdown / winner banner · New heat.</summary>
    public class TurntableHud : MonoBehaviour
    {
        public TurntableEvent ev;
        public string title = "THE 360 TURNTABLE";
        public string subtitle = "Event 12";
        public string version = "v0";
        GUIStyle _big, _mid, _small, _row;

        void OnGUI()
        {
            if (ev == null || ev.spinners.Count == 0) return;
            var cam = Camera.main;
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var a = new Rect(r.x, Screen.height - r.yMax, r.width, r.height);
            float s = Mathf.Clamp(a.width / 540f, 0.6f, 2f), pad = 12 * s;
            _big ??= new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter, wordWrap = true };
            _mid ??= new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold };
            _small ??= new GUIStyle(GUI.skin.label);
            _row ??= new GUIStyle(GUI.skin.label) { richText = true };
            _big.fontSize = Mathf.RoundToInt(52 * s); _mid.fontSize = Mathf.RoundToInt(24 * s);
            _small.fontSize = Mathf.RoundToInt(14 * s); _row.fontSize = Mathf.RoundToInt(14 * s);

            GUI.Label(new Rect(a.x + pad, a.y + pad, a.width * 0.7f, 30 * s), title, _mid);
            GUI.Label(new Rect(a.x + pad, a.y + pad + 28 * s, a.width * 0.7f, 20 * s),
                $"{subtitle} · {ev.turns} turns {(ev.Direction > 0 ? "anticlockwise" : "clockwise")}", _small);
            var tc = new Rect(a.xMax - 180 * s - pad, a.y + pad, 180 * s, 64 * s);
            GUI.Box(tc, GUIContent.none);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 2 * s, tc.width, 30 * s), $"{ev.LiveTime:0.00} s", _mid);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 34 * s, tc.width, 22 * s), $"+{ev.driftPenalty:0.#} s per m drift", _small);

            var box = new Rect(a.x + pad, a.y + 70 * s, 360 * s, (ev.spinners.Count + 1) * 20 * s + 10 * s);
            GUI.Box(box, GUIContent.none);
            GUI.Label(new Rect(box.x + 6 * s, box.y + 4 * s, box.width, 20 * s), "<b>#  spot   result                     STR  LAT  NOISE</b>", _row);
            int i = 0;
            foreach (var x in ev.Standings)
            {
                var p = x.runner;
                string col = x.status is "DQ" or "FELL" ? "#ff7070" : x.place == 1 ? "#ffd84a" : "#ffffff";
                GUI.Label(new Rect(box.x + 6 * s, box.y + (24 + 20 * i) * s, box.width, 20 * s),
                    $"{(x.place > 0 ? x.place : i + 1),-2} {x.name,-5} <color={col}>{ev.Describe(x),-24}</color> {p.strength:0.00}  {p.latencySubsteps}    {p.obsNoise:0.00}", _row);
                i++;
            }
            string banner = ev.Current switch
            {
                TurntableEvent.Phase.Ready => Mathf.CeilToInt(ev.countdownSeconds - ev.PhaseTime).ToString(),
                TurntableEvent.Phase.Result => $"{ev.spinners.First(x => x.place == 1).name} WINS\n{ev.spinners.First(x => x.place == 1).score:0.00}",
                _ => ev.LiveTime < 0.8f ? "SPIN!" : "",
            };
            if (banner.Length > 0) GUI.Label(new Rect(a.x, a.y + a.height * 0.40f, a.width, 150 * s), banner, _big);
            if (GUI.Button(new Rect(a.x + pad, a.yMax - 40 * s - pad, 110 * s, 40 * s), "New heat")) ev.Restart();
            GUI.Label(new Rect(a.xMax - 220 * s - pad, a.yMax - 26 * s - pad, 220 * s, 26 * s), version,
                new GUIStyle(_small) { alignment = TextAnchor.LowerRight });
        }
    }
}
