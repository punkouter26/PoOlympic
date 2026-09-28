using System.Linq;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>Track race HUD (IMGUI placeholder, D2 anchors): title · clock / leader distance · standings with the
    /// event's score (time / top speed / gap) and traits · countdown / winner banner · New heat.</summary>
    public class RaceHud : MonoBehaviour
    {
        public TrackRaceEvent race;
        public string title = "30m DASH";
        public string subtitle = "Event 8";
        public string version = "v0";
        GUIStyle _big, _mid, _small, _row;

        void OnGUI()
        {
            if (race == null || race.runners.Count == 0) return;
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
            GUI.Label(new Rect(a.x + pad, a.y + pad + 28 * s, a.width * 0.7f, 20 * s), subtitle, _small);
            var tc = new Rect(a.xMax - 180 * s - pad, a.y + pad, 180 * s, 64 * s);
            GUI.Box(tc, GUIContent.none);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 2 * s, tc.width, 30 * s), $"{race.LiveTime:0.00} s", _mid);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 34 * s, tc.width, 22 * s), $"leader {race.LeaderX:0.0} / {race.distance:0.#} m", _small);

            var box = new Rect(a.x + pad, a.y + 70 * s, 330 * s, (race.runners.Count + 1) * 20 * s + 10 * s);
            GUI.Box(box, GUIContent.none);
            GUI.Label(new Rect(box.x + 6 * s, box.y + 4 * s, box.width, 20 * s), "<b>#  lane   result            STR  LAT  NOISE</b>", _row);
            int i = 0;
            foreach (var x in race.Standings)
            {
                var p = x.runner;
                string res = race.Current == TrackRaceEvent.Phase.Result || !x.Racing ? race.Describe(x) : $"{x.x:0.0} m  {x.v:0.0} m/s";
                string col = x.status == "DQ" || x.status == "FELL" ? "#ff7070" : x.place == 1 ? "#ffd84a" : "#ffffff";
                GUI.Label(new Rect(box.x + 6 * s, box.y + (24 + 20 * i) * s, box.width, 20 * s),
                    $"{(x.place > 0 ? x.place : i + 1),-2} {x.name,-5} <color={col}>{res,-17}</color> {p.strength:0.00}  {p.latencySubsteps}    {p.obsNoise:0.00}", _row);
                i++;
            }
            string banner = race.Current switch
            {
                TrackRaceEvent.Phase.Ready => Mathf.CeilToInt(race.countdownSeconds - race.PhaseTime).ToString(),
                TrackRaceEvent.Phase.Result => $"{race.runners.First(x => x.place == 1).name} WINS\n{race.Describe(race.runners.First(x => x.place == 1))}",
                _ => race.LiveTime < 0.8f ? "GO!" : "",
            };
            if (banner.Length > 0) GUI.Label(new Rect(a.x, a.y + a.height * 0.40f, a.width, 150 * s), banner, _big);
            if (GUI.Button(new Rect(a.x + pad, a.yMax - 40 * s - pad, 110 * s, 40 * s), "New heat")) race.Restart();
            if (MeetLineup.MenuAvailable && GUI.Button(new Rect(a.x + pad + 120 * s, a.yMax - 40 * s - pad, 90 * s, 40 * s), "Menu")) MeetLineup.ReturnToMenu();
            GUI.Label(new Rect(a.xMax - 220 * s - pad, a.yMax - 26 * s - pad, 220 * s, 26 * s), version,
                new GUIStyle(_small) { alignment = TextAnchor.LowerRight });
        }
    }
}
