using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// D3 tracking cameras (9:16): drives the BroadcastCamera's transform from the event's state.
    ///   Race  — focus = the leader along the race axis, centred across the lanes; live shots cut between trackside,
    ///           head-on and high-wide every shotSeconds
    ///   Arena — focus = centre of the athletes (stationary events); establishing shot, then a slow orbit
    ///   Ready = the establishing shot (the camera as authored) · Result = close-up of the winner.
    /// Letterboxing stays with BroadcastCamera (its target is cleared, so it only letterboxes).
    /// </summary>
    [RequireComponent(typeof(Camera))]
    [DefaultExecutionOrder(210)]
    public class BroadcastDirector : MonoBehaviour
    {
        public enum Kind { Race, Arena }
        public MonoBehaviour board;                     // IBroadcastBoard
        public Kind kind = Kind.Race;
        [Tooltip("Race direction in Unity world space (MuJoCo +x = Unity +x).")]
        public Vector3 forward = Vector3.right;
        [Tooltip("Establishing shot: camera offset from the focus in world space (as authored by the scene builder).")]
        public Vector3 establishing = new(-3.5f, 3.2f, -13f);
        public float lookHeight = 0.9f;
        public float shotSeconds = 5f;
        public float followSharpness = 3f;
        public float orbitSpeed = 0.12f;                // rad/s (arena)
        [Tooltip("Cap on the camera height (m): Event 23 keeps every shot under the trench ceiling.")]
        public float maxCameraHeight = 100f;

        IBroadcastBoard B => board as IBroadcastBoard;
        readonly Dictionary<PolicyRunner, Transform> _pelvis = new();
        float _shotClock;
        int _shot;
        BoardPhase _last = BoardPhase.Ready;
        float _orbit;
        Vector3 _pos, _look;
        bool _init, _cut;

        void Start()
        {
            var cam = GetComponent<BroadcastCamera>();
            if (cam != null) cam.target = null;         // this director owns the framing
        }

        Transform Pelvis(PolicyRunner r)
        {
            if (r == null) return null;
            if (_pelvis.TryGetValue(r, out var t) && t != null) return t;
            var body = FindObjectsByType<MjBody>(FindObjectsSortMode.None).FirstOrDefault(b => b.name == r.athletePrefix + "pelvis");
            _pelvis[r] = body != null ? body.transform : null;
            return _pelvis[r];
        }

        void LateUpdate()
        {
            var b = B;
            if (b == null) return;
            var rows = b.Rows.ToList();
            var bodies = rows.Select(r => (r, t: Pelvis(r.runner))).Where(x => x.t != null && x.r.runner.gameObject.activeInHierarchy).ToList();
            if (bodies.Count == 0) return;
            var f = new Vector3(forward.x, 0, forward.z).normalized;
            var left = Vector3.Cross(Vector3.up, f);    // Unity is left-handed: up × forward = left

            var centre = bodies.Aggregate(Vector3.zero, (s, x) => s + x.t.position) / bodies.Count;
            Vector3 focus;
            if (kind == Kind.Race)
            {
                var lead = bodies.FirstOrDefault(x => !x.r.bad);
                var leadPos = (lead.t != null ? lead.t : bodies[0].t).position;
                // along the race axis: the leader, a little ahead; across: the lane centre (constant)
                focus = centre + f * (Vector3.Dot(leadPos - centre, f) + 1.0f);
            }
            else focus = centre;
            focus.y = lookHeight;

            if (b.BoardState != _last) { _shotClock = 0; _shot = 0; _last = b.BoardState; }
            _shotClock += Time.unscaledDeltaTime;
            Vector3 want, look = focus;
            switch (b.BoardState)
            {
                case BoardPhase.Result:
                    var w = bodies.FirstOrDefault(x => x.r.place == 1);
                    var wp = (w.t != null ? w.t : bodies[0].t).position;
                    look = new Vector3(wp.x, wp.y + 0.2f, wp.z);
                    // front three-quarter close-up: ahead of the winner, off to its right, just above head height
                    want = look + f * 5.5f - left * 3.2f + Vector3.up * 1.4f;
                    break;
                case BoardPhase.Live when kind == Kind.Race:
                    if (_shotClock > shotSeconds) { _shotClock = 0; _shot = (_shot + 1) % 3; _cut = true; }
                    want = _shot switch
                    {
                        1 => focus + f * 9f + Vector3.up * 1.6f - left * 2.0f,          // head-on, looking back at the pack
                        2 => focus - f * 11f + Vector3.up * 7f - left * 6f,            // high and wide from behind
                        _ => focus + establishing,                                    // trackside (establishing offset)
                    };
                    break;
                case BoardPhase.Live:
                    if (_shotClock > shotSeconds * 1.5f) _orbit += orbitSpeed * Time.unscaledDeltaTime;
                    want = focus + Quaternion.AngleAxis(Mathf.Rad2Deg * _orbit, Vector3.up) * establishing;
                    break;
                default:
                    _orbit = 0;
                    want = focus + establishing;
                    break;
            }
            want.y = Mathf.Min(want.y, maxCameraHeight);
            float k = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
            bool cut = !_init || _cut;             // hard cut between live race shots, smooth moves otherwise
            _cut = false;
            _pos = cut ? want : Vector3.Lerp(_pos, want, k);
            _look = cut ? look : Vector3.Lerp(_look, look, k);
            _init = true;
            transform.position = _pos;
            transform.rotation = Quaternion.LookRotation(_look - _pos, Vector3.up);
        }
    }
}
