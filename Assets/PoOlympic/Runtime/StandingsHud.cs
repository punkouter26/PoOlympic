using System.Collections.Generic;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>What an 8-athlete event shows on the StandingsHud.</summary>
    public interface IStandingsBoard
    {
        string SubtitleExtra { get; }
        string ClockLine { get; }
        string InfoLine { get; }
        /// <summary>In display order: place (0 while live), name, result text, highlight (out / DQ), the athlete.</summary>
        IEnumerable<(int place, string name, string result, bool bad, PolicyRunner runner)> Rows { get; }
        /// <summary>Countdown / GO / winner text, empty for none.</summary>
        string Banner { get; }
        void Restart();
    }

    /// <summary>Event HUD (IMGUI placeholder, same layout as RaceHud): title · clock · standings with result and traits ·
    /// countdown / winner banner · New heat. Works for any IStandingsBoard.</summary>
    public class StandingsHud : MonoBehaviour
    {
        public MonoBehaviour board;   // an IStandingsBoard (serializable reference)
        public string title = "EVENT";
        public string subtitle = "Event";
        public string version = "v0";
        GUIStyle _big, _mid, _small, _row;

        void OnGUI()
        {
            if (board is not IStandingsBoard b) return;
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
            GUI.Label(new Rect(a.x + pad, a.y + pad + 28 * s, a.width * 0.7f, 20 * s), $"{subtitle} · {b.SubtitleExtra}", _small);
            var tc = new Rect(a.xMax - 180 * s - pad, a.y + pad, 180 * s, 64 * s);
            GUI.Box(tc, GUIContent.none);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 2 * s, tc.width, 30 * s), b.ClockLine, _mid);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 34 * s, tc.width, 22 * s), b.InfoLine, _small);

            var rows = new List<(int place, string name, string result, bool bad, PolicyRunner runner)>(b.Rows);
            var box = new Rect(a.x + pad, a.y + 70 * s, 380 * s, (rows.Count + 1) * 20 * s + 10 * s);
            GUI.Box(box, GUIContent.none);
            GUI.Label(new Rect(box.x + 6 * s, box.y + 4 * s, box.width, 20 * s), "<b>#  name   result                     STR  LAT  NOISE</b>", _row);
            for (int i = 0; i < rows.Count; i++)
            {
                var x = rows[i];
                string col = x.bad ? "#ff7070" : x.place == 1 ? "#ffd84a" : "#ffffff";
                GUI.Label(new Rect(box.x + 6 * s, box.y + (24 + 20 * i) * s, box.width, 20 * s),
                    $"{(x.place > 0 ? x.place : i + 1),-2} {x.name,-5} <color={col}>{x.result,-26}</color> {x.runner.strength:0.00}  {x.runner.latencySubsteps}    {x.runner.obsNoise:0.00}", _row);
            }
            var banner = b.Banner;
            if (!string.IsNullOrEmpty(banner)) GUI.Label(new Rect(a.x, a.y + a.height * 0.40f, a.width, 150 * s), banner, _big);
            if (GUI.Button(new Rect(a.x + pad, a.yMax - 40 * s - pad, 110 * s, 40 * s), "New heat")) b.Restart();
            GUI.Label(new Rect(a.xMax - 220 * s - pad, a.yMax - 26 * s - pad, 220 * s, 26 * s), version,
                new GUIStyle(_small) { alignment = TextAnchor.LowerRight });
        }
    }
}
