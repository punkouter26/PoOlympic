using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Feature 10 — stadium audio that follows the action (all clips free: Kenney CC0, Gregor Quendel crowd recordings
    /// CC-BY 4.0; see docs/LICENSING.md):
    ///   crowd bed        always-on ambience; volume and pitch rise with TensionMeter.Tension
    ///   tension layer    rhythmic cheering loop faded in as tension climbs (close finishes, athletes wobbling)
    ///   reactions        a crowd reaction on a near fall, a fall and a save; a big cheer for the winner, a bigger one +
    ///                    a chime for a new world record
    ///   impacts          3D body slams / cube hits at the contact, loudness from the MuJoCo contact force
    ///   footsteps        3D, only for the athlete the camera story is about (TensionMeter.Hot), louder with speed
    ///   countdown        a tick per countdown number, a bong at the start call (GO! / HOLD! / SPIN!)
    /// Sources are scene objects (built by the FX upgrade); 3D voices come from a small round-robin pool.
    /// </summary>
    [DefaultExecutionOrder(170)]
    public class ArenaAudio : MonoBehaviour
    {
        public MonoBehaviour board;                    // IBroadcastBoard
        public TensionMeter tension;
        public AudioSource bed, tensionLayer, stinger, ui;
        public AudioSource[] voices;                   // 3D pool

        [Header("Clips")]
        public AudioClip[] cheers, reactions, bodySlams, cubeHits, thuds, footsteps;
        public AudioClip recordChime, goBong, countTick;

        [Header("Mix")]
        [Range(0, 1)] public float master = 0.9f;
        public float bedMin = 0.28f, bedMax = 0.62f;
        public float tensionFrom = 0.4f, tensionFull = 0.85f, tensionMax = 0.55f;
        public float reactionCooldown = 2.5f;

        IBroadcastBoard B => board as IBroadcastBoard;
        int _voice, _heat = -1;
        string _banner = "";
        BoardPhase _phase = BoardPhase.Ready;
        float _lastReaction = -99f;
        readonly System.Random _rng = new(7);
        readonly Dictionary<PolicyRunner, float> _lastStep = new();

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
            float t = tension != null ? tension.Tension : 0.3f;
            float k = 1f - Mathf.Exp(-3f * Time.unscaledDeltaTime);
            if (bed != null)
            {
                bed.volume = Mathf.Lerp(bed.volume, master * Mathf.Lerp(bedMin, bedMax, t), k);
                bed.pitch = Mathf.Lerp(bed.pitch, Mathf.Lerp(0.97f, 1.06f, t), k);
            }
            if (tensionLayer != null)
                tensionLayer.volume = Mathf.Lerp(tensionLayer.volume, master * tensionMax * Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(tensionFrom, tensionFull, t)), k);

            var b = B;
            if (b == null) return;
            if (b.Heat != _heat) { _heat = b.Heat; _banner = ""; }
            // countdown ticks / GO from the banner text ("3", "2", "1", "GO")
            var banner = b.Banner ?? "";
            if (banner != _banner)
            {
                if (banner.Length == 1 && char.IsDigit(banner[0])) Ui(countTick, 0.6f);
                else if (b.BoardState == BoardPhase.Live && banner.Length > 0) Ui(goBong, 0.8f);   // GO! / HOLD! / SPIN!
                _banner = banner;
            }
            if (b.BoardState != _phase)
            {
                if (b.BoardState == BoardPhase.Result) Stinger(Pick(cheers), 0.9f);
                _phase = b.BoardState;
            }
        }

        /// <summary>BroadcastHud: a new world record was set.</summary>
        public void NewRecord()
        {
            Ui(recordChime, 0.9f);
            Stinger(Pick(cheers), 1f);
        }

        void OnNearFall(PolicyRunner r) => React(0.55f);
        void OnFall(PolicyRunner r) => React(0.75f);
        void OnSave(PolicyRunner r) => React(0.7f, cheer: true);

        void React(float volume, bool cheer = false)
        {
            if (Time.unscaledTime - _lastReaction < reactionCooldown) return;
            _lastReaction = Time.unscaledTime;
            Stinger(Pick(cheer ? cheers : reactions), volume * (cheer ? 0.6f : 1f));
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

        void Stinger(AudioClip clip, float volume) { if (clip != null && stinger != null) stinger.PlayOneShot(clip, volume * master); }
        void Ui(AudioClip clip, float volume) { if (clip != null && ui != null) ui.PlayOneShot(clip, volume * master); }
        AudioClip Pick(AudioClip[] clips) => clips == null || clips.Length == 0 ? null : clips[_rng.Next(clips.Length)];
        float Random(float a, float b) => a + (float)_rng.NextDouble() * (b - a);
    }
}
