using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Audio;

namespace PoOlympic
{
    /// <summary>
    /// Feature 10 + GFX/sound idea 8 — stadium audio that follows the action (clips: Kenney CC0, Gregor Quendel crowd
    /// recordings CC-BY 4.0, generated announcer / event SFX; see docs/LICENSING.md):
    ///   mixer            Audio/PoOlympicMix.mixer: Master › Crowd / Sfx / Ui / Announcer; a snapshot per phase (Ready
    ///                    hush, Live, Result swell) blended with "Announce" (crowd + SFX ducked) while the PA talks
    ///   crowd            2D bed + four 3D crowd sectors on the stands (AudioAnchor_Crowd_*): camera cuts change the mix;
    ///                    volume and pitch rise with TensionMeter.Tension, a rhythmic tension layer fades in
    ///   reactions        crowd reaction on a near fall / fall / save; a big cheer for the winner and a new record
    ///   impacts          3D body slams / cube hits at the contact (MuJoCo contact force), the hot athlete's footsteps
    ///   event cues       countdown ticks, starter's gun (races) or whistle (arenas) at GO, buzzer when an athlete is
    ///                    out, air horn at the result, fanfare for the medal ceremony, camera shutters with the flashes
    ///   announcer        PA voice (AudioAnchor_PA_Centre, slap-back echo + the arena reverb zone), chime before the big
    ///                    calls: to your marks / set, lead changes, "is out", what a save, photo finish, "Lane N wins!",
    ///                    new world record, next heat — one line at a time, queued, stale lines dropped
    ///   hit-stop         the listener's low-pass closes while ImpactFx freezes time (70 ms thump)
    /// Sources are scene objects (built by the FX upgrade); 3D voices come from a small round-robin pool.
    /// </summary>
    [DefaultExecutionOrder(170)]
    public class ArenaAudio : MonoBehaviour
    {
        public MonoBehaviour board;                    // IBroadcastBoard
        public TensionMeter tension;
        public CrowdDirector crowd;
        public AudioSource bed, tensionLayer, stinger, ui;
        public AudioSource[] voices;                   // 3D pool
        public AudioSource[] crowdSectors;             // 3D crowd loops on the stands
        public AudioSource pa, eventSfx;
        public AudioLowPassFilter listenerLowPass;

        [Header("Clips")]
        public AudioClip[] cheers, reactions, bodySlams, cubeHits, thuds, footsteps;
        public AudioClip recordChime, goBong, countTick;
        public AudioClip starterGun, whistle, airHorn, buzzer, paChime, fanfare, cameraFlash;

        [Header("Announcer (index = lane - 1)")]
        public AudioClip[] winLines = new AudioClip[8];
        public AudioClip[] outLines = new AudioClip[8];
        public AudioClip[] leadLines = new AudioClip[8];
        public AudioClip recordLine, marksLine, setLine, saveLine, nextHeatLine, resultLine, photoFinishLine;

        [Header("Mixer")]
        public AudioMixer mixer;
        public AudioMixerSnapshot snapReady, snapLive, snapResult, snapAnnounce;

        [Header("Mix")]
        [Range(0, 1)] public float master = 0.9f;
        public float bedMin = 0.28f, bedMax = 0.62f;
        public float tensionFrom = 0.4f, tensionFull = 0.85f, tensionMax = 0.55f;
        public float reactionCooldown = 2.5f;
        public float leadCallCooldown = 7f, saveCallCooldown = 8f;
        [Tooltip("Lead-gap (m) under which a race result is called a photo finish.")]
        public float photoFinishGap = 0.25f;

        IBroadcastBoard B => board as IBroadcastBoard;
        int _voice, _heat = -1;
        string _banner = "";
        BoardPhase _phase = BoardPhase.Ready;
        float _lastReaction = -99f, _lastLead = -99f, _lastSave = -99f, _liveSince, _resultAt = -1f, _nextClick;
        readonly System.Random _rng = new(7);
        readonly Dictionary<PolicyRunner, float> _lastStep = new();
        readonly HashSet<PolicyRunner> _out = new();
        PolicyRunner _leader;
        bool _marksCalled, _setCalled, _winCalled;
        readonly List<(AudioClip clip, bool chime, float until)> _queue = new();
        float _paBusyUntil, _duck;
        AudioMixerSnapshot _phaseSnap;

        void OnEnable()
        {
            AthleteTelemetry.Impact += OnImpact;
            if (tension != null)
            {
                tension.NearFall += OnNearFall;
                tension.Fall += OnFall;
                tension.Save += OnSave;
            }
            if (bed != null && !bed.isPlaying) bed.Play();
            if (tensionLayer != null && !tensionLayer.isPlaying) { tensionLayer.volume = 0f; tensionLayer.Play(); }
            if (crowdSectors != null)
                foreach (var s in crowdSectors)
                    if (s != null && s.clip != null && !s.isPlaying) { s.time = (float)_rng.NextDouble() * s.clip.length; s.Play(); }
            _phaseSnap = snapReady;
        }

        void OnDisable()
        {
            AthleteTelemetry.Impact -= OnImpact;
            if (tension != null)
            {
                tension.NearFall -= OnNearFall;
                tension.Fall -= OnFall;
                tension.Save -= OnSave;
            }
        }

        void Update()
        {
            float now = Time.unscaledTime;
            float t = tension != null ? tension.Tension : 0.3f;
            float k = 1f - Mathf.Exp(-3f * Time.unscaledDeltaTime);
            float bedTarget = master * Mathf.Lerp(bedMin, bedMax, t);
            if (bed != null)
            {
                bed.volume = Mathf.Lerp(bed.volume, bedTarget * (crowdSectors != null && crowdSectors.Length > 0 ? 0.55f : 1f), k);
                bed.pitch = Mathf.Lerp(bed.pitch, Mathf.Lerp(0.97f, 1.06f, t), k);
            }
            if (crowdSectors != null)
                foreach (var s in crowdSectors)
                    if (s != null) { s.volume = Mathf.Lerp(s.volume, bedTarget * 0.9f, k); s.pitch = bed != null ? bed.pitch : 1f; }
            if (tensionLayer != null)
                tensionLayer.volume = Mathf.Lerp(tensionLayer.volume, master * tensionMax * Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(tensionFrom, tensionFull, t)), k);

            HitStopFilter();
            Shutters(now);

            var b = B;
            if (b != null)
            {
                if (b.Heat != _heat) NewHeat(b);
                Countdown(b);
                if (b.BoardState != _phase) PhaseChange(b, now);
                if (b.BoardState == BoardPhase.Live) LiveCalls(b, now);
                if (b.BoardState == BoardPhase.Result && !_winCalled && _resultAt >= 0f && now - _resultAt > 1.3f) WinnerCall(b);
            }
            Announcer(now);
            Mix();
        }

        // ------------------------------------------------------------------------------------------ game flow
        void NewHeat(IBroadcastBoard b)
        {
            bool first = _heat < 0;
            _heat = b.Heat;
            _banner = "";
            _out.Clear();
            _leader = null;
            _marksCalled = _setCalled = _winCalled = false;
            _resultAt = -1f;
            _queue.Clear();
            if (!first) Say(nextHeatLine, chime: true, ttl: 4f);
        }

        void Countdown(IBroadcastBoard b)
        {
            // countdown ticks / GO from the banner text ("3", "2", "1", "GO")
            var banner = b.Banner ?? "";
            if (banner == _banner) return;
            if (banner.Length == 1 && char.IsDigit(banner[0]))
            {
                Ui(countTick, 0.6f);
                if (!_marksCalled) { _marksCalled = true; Say(marksLine, ttl: 2.5f); }
                if (banner == "1" && !_setCalled && Race) { _setCalled = true; Say(setLine, ttl: 1f); }
            }
            else if (b.BoardState == BoardPhase.Live && banner.Length > 0)
            {
                Ui(goBong, 0.45f);                                                 // GO! / HOLD! / SPIN!
                Event(Race ? starterGun : whistle, Race ? 1f : 0.8f);
            }
            _banner = banner;
        }

        bool Race => tension == null || tension.kind == BroadcastDirector.Kind.Race;

        void PhaseChange(IBroadcastBoard b, float now)
        {
            if (b.BoardState == BoardPhase.Live) _liveSince = now;
            if (b.BoardState == BoardPhase.Result)
            {
                Stinger(Pick(cheers), 0.9f);
                Event(airHorn, 0.75f);
                _resultAt = now;
            }
            _phase = b.BoardState;
            _phaseSnap = _phase switch { BoardPhase.Live => snapLive, BoardPhase.Result => snapResult, _ => snapReady };
        }

        void LiveCalls(IBroadcastBoard b, float now)
        {
            var rows = b.Rows.ToList();
            foreach (var r in rows)
                if (r.bad && r.runner != null && _out.Add(r.runner) && now - _liveSince > 0.5f)
                {
                    Event(buzzer, 0.35f);
                    Say(Line(outLines, r.name), ttl: 2.5f);
                }
            if (!Race || rows.Count == 0) return;
            var lead = rows.FirstOrDefault(r => !r.bad).runner;
            if (lead != null && lead != _leader)
            {
                if (_leader != null && now - _liveSince > 2f && now - _lastLead > leadCallCooldown)
                {
                    _lastLead = now;
                    Say(Line(leadLines, rows.First(r => r.runner == lead).name), ttl: 1.5f);
                }
                _leader = lead;
            }
        }

        void WinnerCall(IBroadcastBoard b)
        {
            _winCalled = true;
            var rows = b.Rows.ToList();
            var w = rows.FirstOrDefault(r => r.place == 1);
            if (w.name == null) { Say(resultLine, chime: true, ttl: 3f); return; }
            if (Race && tension != null && !float.IsNaN(tension.LeadGapM) && tension.LeadGapM < photoFinishGap) Say(photoFinishLine, chime: true, ttl: 3f);
            Say(Line(winLines, w.name), chime: !Race || tension == null || float.IsNaN(tension.LeadGapM) || tension.LeadGapM >= photoFinishGap, ttl: 5f);
        }

        static AudioClip Line(AudioClip[] lines, string laneName)
        {
            if (lines == null || string.IsNullOrEmpty(laneName)) return null;
            var digits = new string(laneName.Where(char.IsDigit).ToArray());
            return int.TryParse(digits, out int lane) && lane >= 1 && lane <= lines.Length ? lines[lane - 1] : null;
        }

        /// <summary>BroadcastHud: a new world record was set.</summary>
        public void NewRecord()
        {
            Ui(recordChime, 0.9f);
            Stinger(Pick(cheers), 1f);
            _queue.Clear();
            Say(recordLine, chime: true, ttl: 4f);
            if (crowd != null) crowd.Record();
        }

        /// <summary>PodiumCeremony: the medal ceremony starts.</summary>
        public void Fanfare() => Event(fanfare, 0.8f);

        void OnNearFall(PolicyRunner r) => React(0.55f);
        void OnFall(PolicyRunner r) => React(0.75f);
        void OnSave(PolicyRunner r)
        {
            React(0.7f, cheer: true);
            if (Time.unscaledTime - _lastSave > saveCallCooldown) { _lastSave = Time.unscaledTime; Say(saveLine, ttl: 1.5f); }
        }

        void React(float volume, bool cheer = false)
        {
            if (Time.unscaledTime - _lastReaction < reactionCooldown) return;
            _lastReaction = Time.unscaledTime;
            Stinger(Pick(cheer ? cheers : reactions), volume * (cheer ? 0.6f : 1f));
        }

        // ------------------------------------------------------------------------------------------ announcer
        void Say(AudioClip clip, bool chime = false, float ttl = 3f)
        {
            if (clip == null || pa == null) return;
            if (_queue.Count >= 3) _queue.RemoveAt(0);
            _queue.Add((clip, chime, Time.unscaledTime + ttl));
        }

        void Announcer(float now)
        {
            if (pa == null || now < _paBusyUntil) return;
            _queue.RemoveAll(q => q.until < now);                 // a lead change 5 s late is worse than none
            if (_queue.Count == 0) return;
            var (clip, chime, _) = _queue[0];
            _queue.RemoveAt(0);
            float lead = 0f;
            if (chime && paChime != null)
            {
                pa.PlayOneShot(paChime, 0.55f * master);
                lead = 0.75f;
            }
            if (lead > 0f) { _delayed = clip; _delayedAt = now + lead; }
            else pa.PlayOneShot(clip, master);
            _paBusyUntil = now + lead + clip.length + 0.25f;
        }

        AudioClip _delayed;
        float _delayedAt;

        void LateUpdate()
        {
            if (_delayed != null && Time.unscaledTime >= _delayedAt) { pa.PlayOneShot(_delayed, master); _delayed = null; }
        }

        void Mix()
        {
            float target = Time.unscaledTime < _paBusyUntil ? 1f : 0f;
            _duck = Mathf.MoveTowards(_duck, target, Time.unscaledDeltaTime * (target > _duck ? 5f : 1.5f));
            if (mixer == null || _phaseSnap == null) return;
            if (snapAnnounce != null)
                mixer.TransitionToSnapshots(new[] { _phaseSnap, snapAnnounce }, new[] { 1f - _duck, _duck }, 0.05f);
            else _phaseSnap.TransitionTo(0.3f);
        }

        // ------------------------------------------------------------------------------------------ effects
        void HitStopFilter()
        {
            if (listenerLowPass == null) return;
            bool frozen = Time.timeScale < 0.01f;
            float want = frozen ? 700f : 22000f;
            listenerLowPass.cutoffFrequency = Mathf.Lerp(listenerLowPass.cutoffFrequency, want, frozen ? 1f : 1f - Mathf.Exp(-10f * Time.unscaledDeltaTime));
            listenerLowPass.enabled = listenerLowPass.cutoffFrequency < 21000f;
        }

        void Shutters(float now)
        {
            if (crowd == null || cameraFlash == null || eventSfx == null || crowd.FlashLevel <= 0f || now < _nextClick) return;
            eventSfx.pitch = Random(0.9f, 1.15f);
            eventSfx.PlayOneShot(cameraFlash, 0.18f * master * crowd.FlashLevel);
            _nextClick = now + Random(0.05f, 0.3f) / Mathf.Max(0.3f, crowd.FlashLevel);
        }

        void OnImpact(ImpactEvent e)
        {
            switch (e.kind)
            {
                case ImpactEvent.Kind.FootStrike:
                    if (tension == null || tension.Hot != e.runner || e.speedMps < 0.4f) return;
                    if (_lastStep.TryGetValue(e.runner, out var last) && Time.time - last < 0.18f) return;
                    _lastStep[e.runner] = Time.time;
                    Voice(Pick(footsteps), e.position, Mathf.Lerp(0.25f, 0.7f, e.speedMps / 4f), Random(0.92f, 1.08f));
                    break;
                case ImpactEvent.Kind.BodySlam:
                    float need = e.runner != null && e.runner.crawlSteering ? 2.5f : 1.0f;
                    if (e.weightRatio < need) return;
                    Voice(e.weightRatio > 3f * need ? Pick(bodySlams) : Pick(thuds), e.position,
                          Mathf.Clamp01(0.35f + 0.15f * e.weightRatio / need), Random(0.9f, 1.05f));
                    break;
                case ImpactEvent.Kind.CubeHit:
                    if (e.weightRatio < 0.1f) return;
                    Voice(Pick(cubeHits), e.position, Mathf.Clamp01(0.5f + e.weightRatio), Random(0.95f, 1.1f));
                    break;
            }
        }

        void Voice(AudioClip clip, Vector3 at, float volume, float pitch)
        {
            if (clip == null || voices == null || voices.Length == 0) return;
            var v = voices[_voice];
            _voice = (_voice + 1) % voices.Length;
            v.transform.position = at;
            v.pitch = pitch;
            v.PlayOneShot(clip, volume * master);
        }

        void Event(AudioClip clip, float volume)
        {
            if (clip == null) return;
            var s = eventSfx != null ? eventSfx : stinger;
            if (s == null) return;
            s.pitch = 1f;
            s.PlayOneShot(clip, volume * master);
        }

        void Stinger(AudioClip clip, float volume) { if (clip != null && stinger != null) stinger.PlayOneShot(clip, volume * master); }
        void Ui(AudioClip clip, float volume) { if (clip != null && ui != null) ui.PlayOneShot(clip, volume * master); }
        AudioClip Pick(AudioClip[] clips) => clips == null || clips.Length == 0 ? null : clips[_rng.Next(clips.Length)];
        float Random(float a, float b) => a + (float)_rng.NextDouble() * (b - a);
    }
}
