using System.Linq;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Iron Pedestal heat HUD (D2 anchors, IMGUI placeholder): TL title · TC clock / round / gust strength ·
    /// left column standings (lane, status, out time, traits) · centre countdown / winner banner · BL controls · BR version.
    /// </summary>
    public class HeatHud : MonoBehaviour
    {
        public IronPedestalHeat heat;
        public string title = "IRON PEDESTAL";
        public string subtitle = "Event 1 · last one standing";
        public string version = "v0";
        GUIStyle _big, _mid, _small, _row;

        void OnGUI()
        {
            if (heat == null || heat.runners.Count == 0) return;
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
            var tc = new Rect(a.xMax - 170 * s - pad, a.y + pad, 170 * s, 64 * s);
            GUI.Box(tc, GUIContent.none);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 2 * s, tc.width, 30 * s), $"{heat.LiveTime:0.0} s", _mid);
            GUI.Label(new Rect(tc.x + 8 * s, tc.y + 34 * s, tc.width, 22 * s),
                $"round {heat.Round} · gust {heat.GustNow:0.00} m/s · {heat.StillIn} in", _small);

            // standings
            var box = new Rect(a.x + pad, a.y + 70 * s, 300 * s, (heat.runners.Count + 1) * 20 * s + 10 * s);
            GUI.Box(box, GUIContent.none);
            GUI.Label(new Rect(box.x + 6 * s, box.y + 4 * s, box.width, 20 * s), "<b>#  lane   status          STR  LAT  NOISE</b>", _row);
            int i = 0;
            foreach (var x in heat.Standings)
            {
                var p = x.runner;
                string status = x.In ? (heat.Current == IronPedestalHeat.Phase.Result ? "<color=#ffd84a>WINNER</color>" : "<color=#7CFC7C>IN</color>")
                                     : $"<color=#ff7070>{(x.reason == "STEPPED OFF" ? "OFF" : "FELL")} {x.outAt,5:0.0}s</color>";
                string pos = x.place > 0 ? x.place.ToString() : (i + 1).ToString();
                GUI.Label(new Rect(box.x + 6 * s, box.y + (24 + 20 * i) * s, box.width, 20 * s),
                    $"{pos,-2} {x.name,-6} {status,-22}  {p.strength:0.00}   {p.latencySubsteps}    {p.obsNoise:0.00}", _row);
                i++;
            }

            string banner = heat.Current switch
            {
                IronPedestalHeat.Phase.Ready => Mathf.CeilToInt(heat.countdownSeconds - heat.PhaseTime).ToString(),
                IronPedestalHeat.Phase.Result => $"{string.Join(" & ", heat.runners.Where(x => x.place == 1).Select(x => x.name))} WINS\n{heat.LiveTime:0.0} s",
                _ => heat.LiveTime < 0.8f ? "GO!" : "",
            };
            if (banner.Length > 0) GUI.Label(new Rect(a.x, a.y + a.height * 0.40f, a.width, 150 * s), banner, _big);

            float bw = 110 * s, bh = 40 * s;
            if (GUI.Button(new Rect(a.x + pad, a.yMax - bh - pad, bw, bh), "New heat")) heat.Restart();
            GUI.Label(new Rect(a.xMax - 200 * s - pad, a.yMax - 26 * s - pad, 200 * s, 26 * s), version,
                new GUIStyle(_small) { alignment = TextAnchor.LowerRight });
        }
    }
}
