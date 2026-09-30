using System.Collections.Generic;
using System.Linq;
using Mujoco;
using Unity.Cinemachine;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Broadcast camera director (D3, feature 2: tension-driven Cinemachine cuts). Framing targets — the most relevant
    /// athletes as the heat runs (user, 2026-09-29):
    ///   Race  — focus = the current winner (TensionMeter.Leader), a metre ahead of them, on their lane
    ///   Arena — focus = the current winner (live ranking; in a tie, e.g. last one standing, the steadiest athlete)
    ///   Hot   — an athlete close to falling (TensionMeter danger ≥ hotCutDanger) takes over the shot while in danger
    ///   Winner — place 1 at the result
    /// The director only moves three proxy transforms (focus / hot / winner) and chooses which Cinemachine camera is
    /// live (priority); Cinemachine does the following, damping, blends, hand-held noise and impact shake (impulse):
    ///   Ready   establishing (the authored BroadcastCamera offset)
    ///   Live    calm broadcast pacing (user, 2026-09-29: "too many cuts, closer on the current winner so you can see
    ///           the face"): a front full-body close-up of the current winner (closeupSeconds) alternating with
    ///           one context shot (contextSeconds: trackside / head-on / high-wide for races, the slow orbit for
    ///           arenas); no shot shorter than minShotSeconds; the close-up changes athlete at most every
    ///           leaderSwitchSeconds (a pan, not a cut). A real near fall (danger ≥ hotCutDanger, hotCooldown apart)
    ///           eases to a close-up of that athlete and holds it while the danger lasts
    ///   Result  eases to a front three-quarter close-up of the winner; during the medal ceremony (PodiumCeremony) cuts
    ///           to the fixed podium camera, then back to the winner
    /// Scenes without the Cinemachine rig (wide == null) keep the original direct-transform framing.
    /// Letterboxing stays with BroadcastCamera (its target is cleared, so it only letterboxes).
    /// </summary>
    [RequireComponent(typeof(Camera))]
    [DefaultExecutionOrder(-50)]      // before CinemachineBrain (order 0) reads the proxies
    public class BroadcastDirector : MonoBehaviour
    {
        public enum Kind { Race, Arena }
        public enum Shot { Establishing, Trackside, HeadOn, HighWide, Hot, Winner, Podium, Leader }
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

        [Header("Cinemachine rig (PoOlympic › Broadcast › Upgrade broadcast FX)")]
        public CinemachineCamera wide;
        public CinemachineCamera headOn, high, hot, winner;
        public Transform focusProxy, hotProxy, winnerProxy;
        public TensionMeter tension;
        [Tooltip("Medal ceremony (optional): its podium camera is live while PodiumCeremony.ShowPodium.")]
        public CinemachineCamera podium;
        public PodiumCeremony ceremony;
        [Tooltip("Danger (TensionMeter) at which the director cuts to the athlete in trouble.")]
        public float hotCutDanger = 0.7f;
        public float minShotSeconds = 4f;
        [Tooltip("Seconds after a hot close-up before the next one (unless the danger is extreme, ≥ 0.95).")]
        public float hotCooldown = 8f;
        [Header("Stadium bowl (cameras never go past the track's outer edge into the stands)")]
        [Tooltip("Stadium centre (Stadium.glb AudioAnchor_PA_Centre); unset = no clamp.")]
        public Transform bowlCentre;
        [Tooltip("Unit vector along the stadium's long axis (the straights), world space.")]
        public Vector3 bowlAxis = Vector3.right;
        [Tooltip("Half length of the straights (m) and the largest camera distance from them (track outer edge 47.5 m + 1).")]
        public float bowlHalfStraight = 42.2f, bowlRadius = 48.5f;
        [Header("Leader close-up")]
        public CinemachineCamera leaderClose;
        public Transform leaderProxy;
        public float closeupSeconds = 8f, contextSeconds = 5f, leaderSwitchSeconds = 5f;

        public Shot Current => _shotNow;
        /// <summary>The athlete the live shot is about: the one close to falling (Hot), the winner at the result, else
        /// the current winner of the heat. The HUD stats card follows it.</summary>
        public PolicyRunner Subject { get; private set; }

        IBroadcastBoard B => board as IBroadcastBoard;
        readonly Dictionary<PolicyRunner, Transform> _pelvis = new();
        float _shotClock;
        int _shot;
        BoardPhase _last = BoardPhase.Ready;
        float _orbit;
        Vector3 _pos, _look;
        bool _init, _cut;
        Shot _shotNow = Shot.Establishing;
        float _shotHeld, _hotClock, _hotLeft = -99f;
        PolicyRunner _hotRunner, _closeRunner;
        float _closeSince = -99f;
        CinemachineBrain _brain;
        readonly Dictionary<PolicyRunner, Quaternion> _startRot = new();
        readonly Dictionary<PolicyRunner, Vector3> _faceSmooth = new();

        /// <summary>Where the athlete faces (horizontal): every event starts them facing Unity +x; the pelvis' yaw since
        /// the start of the heat turns that (turntable spins, slalom, backward runners).</summary>
        Vector3 FaceOf(PolicyRunner r)
        {
            var p = Pelvis(r);
            if (p == null) return Vector3.right;
            if (!_startRot.TryGetValue(r, out var q0)) _startRot[r] = q0 = p.rotation;
            var face = p.rotation * Quaternion.Inverse(q0) * Vector3.right;
            face.y = 0f;
            face = face.sqrMagnitude > 0.05f ? face.normalized : Vector3.right;
            // ~1 s smoothing: the pelvis sways with every stride, the camera must not
            if (!_faceSmooth.TryGetValue(r, out var sm)) sm = face;
            sm = Vector3.Slerp(sm, face, 1f - Mathf.Exp(-Time.unscaledDeltaTime / 0.9f));
            _faceSmooth[r] = sm;
            return sm.sqrMagnitude > 0.01f ? sm.normalized : Vector3.right;
        }

        void Start()
        {
            var cam = GetComponent<BroadcastCamera>();
            if (cam != null) cam.target = null;         // this director owns the framing
            _brain = GetComponent<CinemachineBrain>();
            if (tension == null) tension = FindAnyObjectByType<TensionMeter>();
        }

        Vector3 LeaderPosition(List<((int place, string name, string result, bool bad, PolicyRunner runner) r, Transform t)> bodies)
        {
            var leader = tension != null ? tension.Leader : null;
            var lead = bodies.FirstOrDefault(x => x.r.runner == leader && !x.r.bad);
            if (lead.t == null) lead = bodies.FirstOrDefault(x => !x.r.bad);
            return (lead.t != null ? lead.t : bodies[0].t).position;
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
                // the current winner, a little ahead of them, on their own lane
                focus = LeaderPosition(bodies) + f * 1.0f;
            }
            else focus = LeaderPosition(bodies);        // arena: the current winner, not the middle of the group
            focus.y = lookHeight;

            bool phaseChanged = b.BoardState != _last;
            if (phaseChanged)
            {
                _shotClock = 0; _shot = 0; _last = b.BoardState; _snap = true;
                if (b.BoardState == BoardPhase.Ready) { _startRot.Clear(); _faceSmooth.Clear(); }   // re-read the start pose each heat
            }
            _shotClock += Time.unscaledDeltaTime;
            if (wide != null) Direct(b, bodies, focus, f, left);
            else Legacy(b, bodies, focus, f, left);
        }

        // ---------------------------------------------------------------- Cinemachine
        void Direct(IBroadcastBoard b, List<((int place, string name, string result, bool bad, PolicyRunner runner) r, Transform t)> bodies,
                    Vector3 focus, Vector3 f, Vector3 left)
        {
            float k = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
            // a new phase (new heat at the start line, the gun, the result) is a hard cut: no drifting back along the track
            bool snap = !_init || (_snap && b.BoardState != BoardPhase.Result);
            focusProxy.position = snap ? focus : Vector3.Lerp(focusProxy.position, focus, k);
            if (snap) foreach (var c in new[] { wide, headOn, high, hot, winner, leaderClose }) if (c != null) c.PreviousStateIsValid = false;
            _snap = false;
            _init = true;
            _shotHeld += Time.unscaledDeltaTime;

            // who is the story right now
            var hotRunner = tension != null && tension.HotDanger >= hotCutDanger ? tension.Hot : null;
            var w = bodies.FirstOrDefault(x => x.r.place == 1);
            if (w.t != null) winnerProxy.position = w.t.position + Vector3.up * 0.2f;
            Shot want;
            switch (b.BoardState)
            {
                case BoardPhase.Result:
                    want = podium != null && ceremony != null && ceremony.ShowPodium ? Shot.Podium : Shot.Winner;
                    break;
                case BoardPhase.Live:
                {
                    float now = Time.unscaledTime;
                    // the close-up's athlete: the current winner when the close-up starts, kept for the whole shot (a
                    // lead change mid-shot would swing the camera across lanes); between close-ups it follows the
                    // leader at most every leaderSwitchSeconds
                    var lead = tension != null && tension.Leader != null ? tension.Leader : bodies.FirstOrDefault(x => !x.r.bad).r.runner;
                    bool closeOk = _closeRunner != null && bodies.Any(x => x.r.runner == _closeRunner && !x.r.bad);
                    if (!closeOk || (_shotNow != Shot.Leader && lead != null && lead != _closeRunner && now - _closeSince >= leaderSwitchSeconds)) { _closeRunner = lead; _closeSince = now; }
                    bool hotOk = _hotRunner != null && bodies.Any(x => x.r.runner == _hotRunner && !x.r.bad);
                    // no shot shorter than minShotSeconds; the winner close-up is only interrupted after 3/4 of its time
                    bool canCut = _shotHeld >= (_shotNow == Shot.Leader ? Mathf.Max(minShotSeconds, closeupSeconds * 0.75f) : minShotSeconds);
                    if (_shotNow == Shot.Hot && hotOk && (_hotClock < minShotSeconds ||
                        (tension != null && tension.Danger.TryGetValue(_hotRunner, out var dz) && dz > 0.35f && _hotClock < 7f)))
                        want = Shot.Hot;                                     // stay with the athlete in trouble
                    else if (hotRunner != null && _shotNow != Shot.Hot && canCut &&
                             (now - _hotLeft >= hotCooldown || tension.HotDanger >= 0.95f))
                    {
                        want = Shot.Hot;
                        _hotRunner = hotRunner;
                        _hotClock = 0f;
                    }
                    else
                    {
                        // pacing: the start from the side, then close-up of the winner / one context shot, repeat
                        Shot context = kind == Kind.Race ? (_shot % 3) switch { 1 => Shot.HeadOn, 2 => Shot.HighWide, _ => Shot.Trackside } : Shot.Trackside;
                        if (_shotNow == Shot.Establishing) want = Shot.Trackside;
                        else if (_shotNow == Shot.Hot) want = leaderClose != null ? Shot.Leader : context;
                        else if (_shotNow == Shot.Leader) { want = _shotHeld > closeupSeconds ? context : Shot.Leader; if (want != Shot.Leader) _shot++; }
                        else want = _shotHeld > Mathf.Max(contextSeconds, minShotSeconds) && leaderClose != null ? Shot.Leader : _shotNow;
                        if (want == Shot.Leader && _shotNow != Shot.Leader && lead != null) { _closeRunner = lead; _closeSince = now; }   // who is winning now
                    }
                    break;
                }
                default:
                    want = Shot.Establishing;
                    _orbit = 0;
                    break;
            }
            if (want == Shot.Hot)
            {
                _hotClock += Time.unscaledDeltaTime;
                var ht = Pelvis(_hotRunner);
                if (ht != null) hotProxy.position = Vector3.Lerp(hotProxy.position, ht.position, _shotNow == Shot.Hot ? k * 3f : 1f);
            }
            // leader close-up target: chest height (the composer centres it, the face sits in the upper third)
            bool crawl = _closeRunner != null && _closeRunner.crawlSteering;
            var lt = Pelvis(_closeRunner);
            if (leaderProxy != null && lt != null)
            {
                leaderProxy.position = lt.position + Vector3.up * (crawl ? 0.1f : 0.05f);   // mid-body, attached (a lag = a runner off-frame)
            }

            // offsets (world space, relative to the proxies)
            if (kind == Kind.Arena && b.BoardState == BoardPhase.Live)
            {
                if (_shotClock > shotSeconds * 1.5f) _orbit += orbitSpeed * Time.unscaledDeltaTime;
                SetOffset(wide, focusProxy, Quaternion.AngleAxis(Mathf.Rad2Deg * _orbit, Vector3.up) * establishing);
            }
            else SetOffset(wide, focusProxy, establishing);
            SetOffset(headOn, focusProxy, f * 9f + Vector3.up * 1.6f - left * 2.0f);
            SetOffset(high, focusProxy, -f * 11f + Vector3.up * 7f - left * 6f);
            bool low = _hotRunner != null && _hotRunner.crawlSteering;
            var hf = _hotRunner != null ? FaceOf(_hotRunner) : f;
            SetOffset(hot, hotProxy, hf * 3.2f + Vector3.Cross(Vector3.up, hf) * 2.8f + Vector3.up * (low ? 0.6f : 1.0f));   // front three-quarter, full body
            if (leaderClose != null && _closeRunner != null)
            {
                var lf = FaceOf(_closeRunner);
                // in front, a little to the side, from chest height: 3.2 m with a 42° lens = the whole body, face visible
                // (user, 2026-09-29: "back up about 2 feet so the whole body is in picture")
                SetOffset(leaderClose, leaderProxy, lf * 3.2f + Vector3.Cross(Vector3.up, lf) * 1.0f + Vector3.up * (crawl ? 0.4f : 0.45f));
            }
            SetOffset(winner, winnerProxy, f * 5.5f - left * 3.2f + Vector3.up * 1.2f);

            if (want != _shotNow || !_liveSet) Activate(want);
            Subject = want == Shot.Hot ? _hotRunner
                    : want == Shot.Winner || want == Shot.Podium ? w.r.runner
                    : _closeRunner != null ? _closeRunner : bodies.FirstOrDefault(x => !x.r.bad).r.runner;
        }

        bool _liveSet, _snap;

        void Activate(Shot s)
        {
            var from = _shotNow;
            if (from == Shot.Hot) _hotLeft = Time.unscaledTime;
            _shotNow = s;
            _shotHeld = 0f;
            _liveSet = true;
            if (_brain != null)
            {
                _brain.DefaultBlend = s switch
                {
                    Shot.Hot => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, 0.6f),
                    Shot.Winner when from == Shot.Podium => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.Cut, 0f),
                    Shot.Winner => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, 1.1f),
                    _ when from == Shot.Hot => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.EaseInOut, 0.7f),
                    _ => new CinemachineBlendDefinition(CinemachineBlendDefinition.Styles.Cut, 0f),   // broadcast cuts
                };
            }
            var live = CameraOf(s);
            foreach (var c in new[] { wide, headOn, high, hot, winner, podium, leaderClose })
                if (c != null) c.Priority = c == live ? 20 : 10;
        }

        CinemachineCamera CameraOf(Shot s) => s switch
        {
            Shot.HeadOn => headOn,
            Shot.HighWide => high,
            Shot.Hot => hot,
            Shot.Winner => winner,
            Shot.Podium => podium,
            Shot.Leader => leaderClose,
            _ => wide,
        };

        /// <summary>Pull a camera position back inside the stadium curve (distance ≤ bowlRadius from the centre segment of
        /// the straights): outer-lane shots used to end up in the crowd (user, 2026-09-29).</summary>
        public Vector3 ClampToBowl(Vector3 p)
        {
            if (bowlCentre == null) return p;
            var c = bowlCentre.position;
            var axis = new Vector3(bowlAxis.x, 0f, bowlAxis.z).normalized;
            var v = new Vector3(p.x - c.x, 0f, p.z - c.z);
            float along = Mathf.Clamp(Vector3.Dot(v, axis), -bowlHalfStraight, bowlHalfStraight);
            var d = v - axis * along;
            float dist = d.magnitude;
            if (dist <= bowlRadius) return p;
            var q = axis * along + d / dist * bowlRadius;
            return new Vector3(c.x + q.x, p.y, c.z + q.z);
        }

        void SetOffset(CinemachineCamera cam, Transform target, Vector3 offset)
        {
            if (cam == null) return;
            offset.y = Mathf.Min(offset.y, maxCameraHeight - target.position.y);
            offset = ClampToBowl(target.position + offset) - target.position;
            var follow = cam.GetComponent<CinemachineFollow>();
            if (follow != null) follow.FollowOffset = offset;
        }

        // ---------------------------------------------------------------- original direct framing (no rig)
        void Legacy(IBroadcastBoard b, List<((int place, string name, string result, bool bad, PolicyRunner runner) r, Transform t)> bodies,
                    Vector3 focus, Vector3 f, Vector3 left)
        {
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
