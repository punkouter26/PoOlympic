using System.Linq;
using Mujoco;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>
    /// GFX idea 2 — live stadium screens. The two scoreboards and the centre-hung video cube (Screen_Live in
    /// Stadium.glb) show one render texture, drawn by a UI Toolkit panel (PanelSettings.targetTexture) at 1280 x 448:
    ///   [ standings top 4 | live camera feed 840 px (the athlete the story is about) | event · clock · heat ]
    /// The cube's material shows only the centre 840 px (its faces are 16:8.5), the scoreboards the whole board.
    /// The feed camera renders into its own small texture every `camEvery` frames (a second view of the arena costs
    /// draw calls; 20 fps on a screen is plenty). Scene objects are built by the editor (StadiumShowcase); render only.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    [DefaultExecutionOrder(180)]
    public class ScreenFeed : MonoBehaviour
    {
        public MonoBehaviour board;                    // IBroadcastBoard
        public TensionMeter tension;
        public Camera feedCamera;
        public RenderTexture feedTexture;
        public StyleSheet style;
        public string title = "EVENT";
        public string subtitle = "";
        public Vector3 forward = Vector3.right;
        [Range(1, 6)] public int camEvery = 3;

        IBroadcastBoard B => board as IBroadcastBoard;
        VisualElement _root, _rows, _banner;
        Label _clock, _info, _name, _tag, _bannerText;
        readonly Label[] _place = new Label[4], _lane = new Label[4], _result = new Label[4];
        float _bannerUntil;
        Vector3 _camPos, _look;
        bool _camInit;
        PolicyRunner _subject;
        BroadcastDirector _director;
        Transform _subjectPelvis;

        void OnEnable()
        {
            var doc = GetComponent<UIDocument>();
            _root = doc.rootVisualElement;
            _root.Clear();
            _root.StretchToParentSize();          // a render-texture panel's document root does not fill it by itself
            if (style != null) _root.styleSheets.Add(style);
            var frame = Add(_root, "sb-frame");
            var left = Add(frame, "sb-left");
            Text(left, "sb-head", "STANDINGS");
            _rows = Add(left, "sb-rows");
            for (int i = 0; i < 4; i++)
            {
                var r = Add(_rows, "sb-row");
                _place[i] = Text(r, "sb-place", "");
                _lane[i] = Text(r, "sb-lane", "");
                _result[i] = Text(r, "sb-result", "");
            }
            var cam = Add(frame, "sb-cam");
            var img = new Image { image = feedTexture, scaleMode = ScaleMode.ScaleAndCrop };
            img.AddToClassList("sb-feed");
            cam.Add(img);
            Text(cam, "sb-brand", "POOLYMPICS  ·  LIVE");
            var bar = Add(cam, "sb-bar");
            _name = Text(bar, "sb-name", "");
            _tag = Text(bar, "sb-tag", "");
            _banner = Add(cam, "sb-banner");
            _bannerText = Text(_banner, "sb-banner-text", "WORLD RECORD");
            _banner.style.display = DisplayStyle.None;
            var right = Add(frame, "sb-right");
            Text(right, "sb-title", title);
            Text(right, "sb-sub", subtitle);
            _clock = Text(right, "sb-clock", "");
            _info = Text(right, "sb-info", "");
        }

        static VisualElement Add(VisualElement p, string cls) { var v = new VisualElement(); v.AddToClassList(cls); p.Add(v); return v; }
        static Label Text(VisualElement p, string cls, string t) { var l = new Label(t); l.AddToClassList(cls); p.Add(l); return l; }

        /// <summary>A big banner over the feed for `seconds` (world record, winner).</summary>
        public void Banner(string text, float seconds)
        {
            _bannerText.text = text;
            _bannerUntil = Time.unscaledTime + seconds;
        }

        void LateUpdate()
        {
            var b = B;
            if (b == null || _root == null) return;
            var rows = b.Rows.ToList();
            for (int i = 0; i < 4; i++)
            {
                bool on = i < rows.Count;
                _place[i].parent.style.visibility = on ? Visibility.Visible : Visibility.Hidden;
                if (!on) continue;
                var r = rows[i];
                _place[i].text = (r.place > 0 ? r.place : i + 1).ToString();
                _lane[i].text = r.name;
                _result[i].text = r.result;
                _place[i].parent.EnableInClassList("sb-row-bad", r.bad);
                _place[i].parent.EnableInClassList("sb-row-first", r.place == 1);
            }
            _clock.text = b.ClockLine;
            _info.text = b.InfoLine;
            bool banner = Time.unscaledTime < _bannerUntil;
            _banner.style.display = banner ? DisplayStyle.Flex : DisplayStyle.None;
            if (banner) _banner.style.opacity = 0.75f + 0.25f * Mathf.Sin(Time.unscaledTime * 8f);

            // who the feed follows: the winner at the result, an athlete in trouble, else the leader
            PolicyRunner who;
            string tag;
            if (b.BoardState == BoardPhase.Result) { who = rows.FirstOrDefault(r => r.place == 1).runner; tag = "WINNER"; }
            else
            {
                // the same story as the broadcast camera (BroadcastDirector.Subject)
                if (_director == null) _director = FindAnyObjectByType<BroadcastDirector>();
                who = _director != null && _director.Subject != null ? _director.Subject
                    : tension != null && tension.Leader != null ? tension.Leader : rows.FirstOrDefault(r => !r.bad).runner;
                tag = _director != null && _director.Current == BroadcastDirector.Shot.Hot ? "IN TROUBLE" : b.BoardState == BoardPhase.Live ? "LEADER" : "ON THE LINE";
            }
            if (who != null)
            {
                _name.text = rows.FirstOrDefault(r => r.runner == who).name + "  " + (Odds.IsZombie(who) ? "ZOMBIE" : "MATT");
                _tag.text = tag;
            }
            Follow(who);
            if (feedCamera != null) feedCamera.enabled = Time.frameCount % Mathf.Max(1, camEvery) == 0;
        }

        void Follow(PolicyRunner who)
        {
            if (feedCamera == null || who == null) return;
            if (who != _subject)
            {
                _subject = who;
                _subjectPelvis = FindObjectsByType<MjBody>().FirstOrDefault(x => x.name == who.athletePrefix + "pelvis")?.transform;
                _camInit = false;
            }
            if (_subjectPelvis == null) return;
            var f = new Vector3(forward.x, 0, forward.z).normalized;
            var left = Vector3.Cross(Vector3.up, f);
            var p = _subjectPelvis.position;
            var want = p + f * 3.2f - left * 2.0f + Vector3.up * 0.7f;
            var look = p + Vector3.up * 0.15f;
            float k = _camInit ? 1f - Mathf.Exp(-4f * Time.unscaledDeltaTime) : 1f;
            _camPos = Vector3.Lerp(_camPos, want, k);
            _look = Vector3.Lerp(_look, look, k);
            _camInit = true;
            feedCamera.transform.SetPositionAndRotation(_camPos, Quaternion.LookRotation(_look - _camPos, Vector3.up));
        }
    }
}
