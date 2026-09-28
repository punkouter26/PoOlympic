using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic
{
    /// <summary>
    /// Stadium hub: all 30 event venues in one scene. IMGUI event picker (placeholder for the broadcast UI) that flies the
    /// camera to a venue, shows its rules and 8 competitor spots, and loads the event scene when it is built.
    /// </summary>
    public class StadiumDirector : MonoBehaviour
    {
        public Camera cam;
        public EventVenue[] venues = new EventVenue[0];
        public int selected = 1;
        public Vector3 viewOffset = new(-9f, 9f, -17f);   // from the home-straight side, elevated
        public float flySharpness = 3f;
        static readonly string[] PhaseNames = { "", "Stability & Balance", "Fundamental Track & Gait", "Omnidirectional Agility",
                                                "High Impact & Jumping", "Heavy Athletics & Transitions", "The Extreme Decathlon" };
        GUIStyle _title, _body, _small;

        EventVenue Current => System.Array.Find(venues, v => v != null && v.number == selected);

        void Start()
        {
            var v = Current;
            if (v != null && cam != null) { cam.transform.position = v.Centre + viewOffset; cam.transform.LookAt(v.Centre + Vector3.up); }
        }

        void LateUpdate()
        {
            var v = Current;
            if (v == null || cam == null) return;
            var target = v.Centre;
            float k = 1f - Mathf.Exp(-flySharpness * Time.unscaledDeltaTime);
            cam.transform.position = Vector3.Lerp(cam.transform.position, target + viewOffset, k);
            var want = Quaternion.LookRotation(target + Vector3.up - cam.transform.position, Vector3.up);
            cam.transform.rotation = Quaternion.Slerp(cam.transform.rotation, want, k);
        }

        void OnGUI()
        {
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var a = new Rect(r.x, Screen.height - r.yMax, r.width, r.height);
            float s = Mathf.Clamp(a.width / 540f, 0.6f, 2f), pad = 10 * s;
            _title ??= new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold, wordWrap = true };
            _body ??= new GUIStyle(GUI.skin.label) { wordWrap = true };
            _small ??= new GUIStyle(GUI.skin.label);
            _title.fontSize = Mathf.RoundToInt(22 * s);
            _body.fontSize = Mathf.RoundToInt(15 * s);
            _small.fontSize = Mathf.RoundToInt(13 * s);

            GUI.Label(new Rect(a.x + pad, a.y + pad, a.width, 30 * s), "POOLYMPICS · 30 EVENTS", _title);
            // event grid: 6 columns x 5 rows
            float bw = (a.width - 2 * pad - 5 * 4 * s) / 6f, bh = 30 * s;
            for (int i = 0; i < 30; i++)
            {
                int n = i + 1, col = i % 6, row = i / 6;
                var v = System.Array.Find(venues, x => x != null && x.number == n);
                var rect = new Rect(a.x + pad + col * (bw + 4 * s), a.y + pad + 36 * s + row * (bh + 4 * s), bw, bh);
                var prev = GUI.backgroundColor;
                GUI.backgroundColor = n == selected ? Color.yellow : (v != null && v.Playable ? new Color(0.5f, 1f, 0.5f) : Color.white);
                if (GUI.Button(rect, n.ToString("00"))) selected = n;
                GUI.backgroundColor = prev;
            }
            var cur = Current;
            if (cur == null) return;
            float y0 = a.y + pad + 36 * s + 5 * (bh + 4 * s) + 6 * s;
            var card = new Rect(a.x + pad, y0, a.width - 2 * pad, 170 * s);
            GUI.Box(card, GUIContent.none);
            GUI.Label(new Rect(card.x + 8 * s, card.y + 6 * s, card.width - 16 * s, 30 * s), $"{cur.number:00} · {cur.eventName}", _title);
            GUI.Label(new Rect(card.x + 8 * s, card.y + 36 * s, card.width - 16 * s, 20 * s),
                $"Phase {cur.phase} — {PhaseNames[Mathf.Clamp(cur.phase, 0, 6)]} · {cur.skill}", _small);
            GUI.Label(new Rect(card.x + 8 * s, card.y + 58 * s, card.width - 16 * s, 70 * s), cur.rules, _body);
            GUI.Label(new Rect(card.x + 8 * s, card.y + 132 * s, card.width * 0.6f, 24 * s),
                $"8 competitor spots · brain: {cur.brain} · {(cur.Playable ? "PLAYABLE" : "coming soon")}", _small);
            if (cur.Playable && GUI.Button(new Rect(card.xMax - 110 * s, card.y + 128 * s, 100 * s, 32 * s), "Play ▸"))
                SceneManager.LoadScene(cur.eventScene);
            if (GUI.Button(new Rect(a.x + pad, a.yMax - 40 * s - pad, 60 * s, 40 * s), "◀")) selected = (selected + 28) % 30 + 1;
            if (GUI.Button(new Rect(a.xMax - 60 * s - pad, a.yMax - 40 * s - pad, 60 * s, 40 * s), "▶")) selected = selected % 30 + 1;
        }
    }
}
