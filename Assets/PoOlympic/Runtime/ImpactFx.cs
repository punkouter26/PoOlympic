using System.Collections;
using System.Collections.Generic;
using Unity.Cinemachine;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.VFX;

namespace PoOlympic
{
    /// <summary>
    /// Feature 3 — impact VFX, camera shake and hit-stop, driven by the MuJoCo contact forces of AthleteTelemetry:
    ///   foot strike  dust puff scaled by running speed (runners above footDustMinSpeed; one puff per athlete per 0.2 s)
    ///   body slam    a non-foot body part lands on a surface with ≥ slamMinWeights body weights → dust ring + shockwave;
    ///                big slams also shake the camera (Cinemachine impulse) and freeze the frame (hit-stop)
    ///   cube hit     a pool cube strikes an athlete → sparks (VFX Graph; Shuriken fallback without compute shaders),
    ///                shake + hit-stop above cubeHitStopWeights
    /// Hit-stop (not while a heat is live, hitStopWhileLive — the race never slows down) = Time.timeScale 0 for
    /// hitStopSeconds of real time: physics (FixedUpdate) simply pauses, fixedDeltaTime
    /// is untouched, so the simulation and parity are unchanged. Every effect instance is pooled in the scene (built by
    /// PoOlympic › Broadcast › Upgrade broadcast FX); nothing is instantiated at runtime.
    /// </summary>
    public class ImpactFx : MonoBehaviour
    {
        [Header("Pools (scene instances)")]
        public ParticleSystem[] footDust;
        public ParticleSystem[] slamDust;
        public ParticleSystem[] shockwaves;
        public VisualEffect[] sparks;
        public ParticleSystem[] sparksFallback;
        [Tooltip("Ground marks (flat, long-lived): a scuff where a body slams down, a trail under a body part that slides.")]
        public ParticleSystem[] marks;
        [Tooltip("Projected scuffs (URP decals: they wrap the pedestal edge and slopes). Used instead of `marks` when present.")]
        public DecalProjector[] decals;
        public float decalSeconds = 18f;
        [Tooltip("Global volume (weight 0 at rest) flashed by big slams: colour fringe + darker corners, no lens warp. " +
                 "Player option: GameSettings.FocusBlur (off by default on phones).")]
        public Volume pulseVolume;
        public float pulseSeconds = 0.18f;
        public CinemachineImpulseSource impulse;

        [Header("Force scaling")]
        [Tooltip("A full-strength hit emits this many times the particles of the prefab's burst, this much faster.")]
        public float emitBoost = 2.5f;
        public float speedBoost = 1.8f;
        [Tooltip("Dust colour per surface (multiplies the prefab's colour): track, metal, glass.")]
        public Color[] surfaceTint = { Color.white, new Color(0.75f, 0.8f, 0.9f, 0.45f), new Color(0.9f, 0.97f, 1f, 0.35f) };

        [Header("Thresholds")]
        public float footDustMinSpeed = 1.2f;
        [Tooltip("Body-slam effects from this contact force (body weights).")]
        public float slamMinWeights = 1.2f;
        [Tooltip("Crawlers put hands and knees down all the time: their slams need this many times the force.")]
        public float crawlerSlamFactor = 2.5f;
        public float cubeMinWeights = 0.15f;
        [Tooltip("Shake + hit-stop from this slam force (body weights).")]
        public float bigSlamWeights = 3f;
        public float cubeHitStopWeights = 0.6f;
        [Tooltip("A body part sliding on the ground faster than this (m/s) leaves scuff marks, one per slideMarkGap metres.")]
        public float slideMarkSpeed = 1.5f, slideMarkGap = 0.35f;

        [Header("Hit-stop")]
        public bool hitStop = true;
        public float hitStopSeconds = 0.07f;
        public float hitStopCooldown = 2.5f;
        [Tooltip("Freeze frames while a heat is live. Off (user, 2026-09-29: \"no slo motion during the race\"): the " +
                 "live heat runs at full speed; shake and effects still play.")]
        public bool hitStopWhileLive;

        public int Emitted { get; private set; }

        int _dust, _slam, _wave, _spark, _mark, _decal;
        float[] _decalBorn;
        float _pulse;
        struct Base { public float speedMin, speedMax; public int count; public Color color; }
        readonly Dictionary<ParticleSystem, Base> _base = new();
        readonly Dictionary<AthleteTelemetry, Vector3> _lastMark = new();
        float _lastStop = -99f;
        bool _stopping;
        float _stopUntil;
        IBroadcastBoard _board;
        bool _boardSearched;
        bool _vfxOk;
        readonly Dictionary<PolicyRunner, float> _lastPuff = new();

        void OnEnable()
        {
            _vfxOk = SystemInfo.supportsComputeShaders && sparks != null && sparks.Length > 0;
            AthleteTelemetry.Impact += OnImpact;
        }

        void OnDisable()
        {
            AthleteTelemetry.Impact -= OnImpact;
            if (_stopping) { Time.timeScale = GameSettings.SimSpeed; _stopping = false; }
        }

        void OnImpact(ImpactEvent e)
        {
            switch (e.kind)
            {
                case ImpactEvent.Kind.FootStrike:
                    if (e.speedMps < footDustMinSpeed || footDust == null || footDust.Length == 0) return;
                    if (_lastPuff.TryGetValue(e.runner, out var t) && Time.time - t < 0.2f) return;
                    _lastPuff[e.runner] = Time.time;
                    Burst(footDust, ref _dust, e.position + Vector3.up * 0.02f, Mathf.Clamp01((e.speedMps - footDustMinSpeed) / 3f), e.surface, true);
                    break;
                case ImpactEvent.Kind.BodySlam:
                    float need = slamMinWeights * (e.runner != null && e.runner.crawlSteering ? crawlerSlamFactor : 1f);
                    if (e.weightRatio < need) return;
                    float s = Mathf.Clamp01((e.weightRatio - need) / (bigSlamWeights * 2f));
                    Haptics.Pulse(s);
                    Burst(slamDust, ref _slam, e.position + Vector3.up * 0.03f, s, e.surface, true);
                    Burst(shockwaves, ref _wave, new Vector3(e.position.x, e.position.y + 0.02f, e.position.z), s);
                    Mark(e.position, e.normal, s);
                    if (e.weightRatio >= bigSlamWeights * (need / slamMinWeights)) Big(e, 0.25f + 0.5f * s);
                    break;
                case ImpactEvent.Kind.CubeHit:
                    if (e.weightRatio < cubeMinWeights) return;
                    float c = Mathf.Clamp01(e.weightRatio / (cubeHitStopWeights * 3f));
                    Haptics.Pulse(c);
                    Sparks(e.position, e.normal, c);
                    if (e.weightRatio >= cubeHitStopWeights) Big(e, 0.3f + 0.5f * c);
                    break;
            }
        }

        /// <param name="scaled">dust: the contact force also sets how many particles fly and how fast, and the
        /// surface tints them (rings and marks are one particle each: size only)</param>
        void Burst(ParticleSystem[] pool, ref int next, Vector3 at, float strength, Surface surface = Surface.Track, bool scaled = false)
        {
            if (pool == null || pool.Length == 0) return;
            var ps = pool[next];
            next = (next + 1) % pool.Length;
            if (ps == null) return;
            ps.transform.position = at;
            ps.transform.localScale = Vector3.one * Mathf.Lerp(0.8f, 1.6f, strength);
            if (!scaled) { ps.Play(true); Emitted++; return; }
            var main = ps.main;
            if (!_base.TryGetValue(ps, out var b))
            {
                var em = ps.emission;
                b = new Base
                {
                    speedMin = main.startSpeed.constantMin, speedMax = main.startSpeed.constantMax, color = main.startColor.color,
                    count = em.burstCount > 0 ? (int)em.GetBurst(0).count.constant : 0,
                };
                _base[ps] = b;
            }
            float k = Mathf.Lerp(1f, speedBoost, strength);
            main.startSpeed = new ParticleSystem.MinMaxCurve(b.speedMin * k, b.speedMax * k);
            main.startColor = b.color * surfaceTint[Mathf.Min((int)surface, surfaceTint.Length - 1)];
            ps.Play(true);
            ps.Emit(Mathf.RoundToInt(b.count * (emitBoost - 1f) * strength));      // on top of the prefab's own burst
            Emitted++;
        }

        /// <summary>Ground mark where a body lands or drags: a projected decal (wraps edges and slopes), else the flat
        /// particle of the older scenes.</summary>
        void Mark(Vector3 at, Vector3 normal, float strength)
        {
            if (!GameSettings.ImpactMarks) return;
            if (normal.sqrMagnitude > 0.01f && normal.y < 0.5f) return;      // a hit against a pole or a wall: nothing to mark
            if (decals == null || decals.Length == 0) { Burst(marks, ref _mark, at + Vector3.up * 0.012f, strength); return; }
            var d = decals[_decal];
            if (d == null) return;
            _decalBorn ??= new float[decals.Length];
            _decalBorn[_decal] = Time.time;
            _decal = (_decal + 1) % decals.Length;
            if (normal.sqrMagnitude < 0.01f) normal = Vector3.up;
            // projects along its +z, into the surface; a random roll so no two scuffs line up
            d.transform.SetPositionAndRotation(at, Quaternion.LookRotation(-normal, Vector3.forward) * Quaternion.Euler(0f, 0f, Random.value * 360f));
            float w = Mathf.Lerp(0.8f, 1.6f, strength);
            d.transform.localScale = new Vector3(w, w, 1f);
            d.fadeFactor = 1f;
            d.enabled = true;
            Emitted++;
        }

        /// <summary>Decals fade over the last 30 % of their life and switch off (a disabled projector costs nothing).</summary>
        void FadeDecals()
        {
            if (_decalBorn == null) return;
            for (int i = 0; i < decals.Length; i++)
            {
                var d = decals[i];
                if (d == null || !d.enabled) continue;
                float age = (Time.time - _decalBorn[i]) / Mathf.Max(0.1f, decalSeconds);
                if (age >= 1f || !GameSettings.ImpactMarks) d.enabled = false;
                else d.fadeFactor = Mathf.Clamp01((1f - age) / 0.3f);
            }
        }

        void Sparks(Vector3 at, Vector3 normal, float strength)
        {
            if (_vfxOk)
            {
                var v = sparks[_spark];
                _spark = (_spark + 1) % sparks.Length;
                if (v == null) return;
                v.transform.position = at;
                v.transform.rotation = Quaternion.LookRotation(normal.sqrMagnitude > 0.01f ? normal : Vector3.up);
                if (v.HasVector3("Initial Velocity")) v.SetVector3("Initial Velocity", normal * (2f + 4f * strength) + Vector3.up * 1.5f);
                v.Reinit();
                v.Play();
                StartCoroutine(StopSoon(v, 0.12f));
                Emitted++;
            }
            else
            {
                Burst(sparksFallback, ref _spark, at, strength);
                if (sparksFallback != null && sparksFallback.Length > 0) _spark %= sparksFallback.Length;
            }
        }

        static IEnumerator StopSoon(VisualEffect v, float after)
        {
            yield return new WaitForSecondsRealtime(after);
            if (v != null) v.Stop();                     // stop spawning; live sparks finish their life
        }

        void Big(ImpactEvent e, float shake)
        {
            if (impulse != null) impulse.GenerateImpulseAt(e.position, (Vector3.down + e.normal * 0.3f) * shake);
            _pulse = Mathf.Max(_pulse, shake);
            if (!hitStop || _stopping || Time.unscaledTime - _lastStop < hitStopCooldown) return;
            if (!hitStopWhileLive && HeatLive()) return;                // no slow motion during the race
            _stopping = true;
            _lastStop = Time.unscaledTime;
            _stopUntil = Time.realtimeSinceStartup + hitStopSeconds;
            Time.timeScale = 0f;
        }

        // Hit-stop release in Update (not a WaitForSecondsRealtime coroutine: in the editor's single-step capture mode
        // that coroutine never resumed and left the heat frozen at timeScale 0).
        /// <summary>The scene's event board is in its live phase (no board: treated as live).</summary>
        bool HeatLive()
        {
            if (!_boardSearched)
            {
                _boardSearched = true;
                foreach (var mb in FindObjectsByType<MonoBehaviour>())
                    if (mb is IBroadcastBoard b) { _board = b; break; }
            }
            return _board == null || _board.BoardState == BoardPhase.Live;
        }

        /// <summary>Scuff trail: a body part dragged along the ground (a fall that slides, a crawler's knees).</summary>
        void SlideMarks()
        {
            if (!GameSettings.ImpactMarks) return;
            foreach (var t in AthleteTelemetry.All)
            {
                if (t == null || !t.Ready || t.SlideSpeed < slideMarkSpeed) continue;
                if (_lastMark.TryGetValue(t, out var last) && (t.SlidePoint - last).sqrMagnitude < slideMarkGap * slideMarkGap) continue;
                _lastMark[t] = t.SlidePoint;
                Mark(t.SlidePoint, Vector3.up, 0f);
            }
        }

        void Update()
        {
            SlideMarks();
            FadeDecals();
            // real time: the flash runs through a hit-stop
            _pulse = Mathf.Max(0f, _pulse - Time.unscaledDeltaTime / Mathf.Max(0.02f, pulseSeconds));
            if (pulseVolume != null) pulseVolume.weight = GameSettings.FocusBlur ? Mathf.Clamp01(_pulse) : 0f;
            if (!_stopping || Time.realtimeSinceStartup < _stopUntil) return;
            Time.timeScale = GameSettings.SimSpeed;       // the viewing speed chosen meanwhile, not the one before the stop
            _stopping = false;
        }
    }

    /// <summary>Phone vibration scaled by an impact's strength (Android VibrationEffect: length and amplitude); no-op in
    /// the editor and on desktop. Player option: GameSettings.Haptics.</summary>
    public static class Haptics
    {
        static float _last = -1f;
#if UNITY_ANDROID && !UNITY_EDITOR
        static AndroidJavaObject _vibrator;
        static AndroidJavaClass _effect;
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => _last = -1f;

        /// <param name="strength">0 = a tick (15 ms, soft) … 1 = a thump (60 ms, full amplitude)</param>
        public static void Pulse(float strength)
        {
            if (!GameSettings.Haptics || Time.unscaledTime - _last < 0.1f) return;
            _last = Time.unscaledTime;
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                if (_vibrator == null)
                {
                    using var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
                    using var activity = player.GetStatic<AndroidJavaObject>("currentActivity");
                    _vibrator = activity.Call<AndroidJavaObject>("getSystemService", "vibrator");
                    _effect = new AndroidJavaClass("android.os.VibrationEffect");
                }
                using var fx = _effect.CallStatic<AndroidJavaObject>("createOneShot", (long)Mathf.Lerp(15f, 60f, strength), Mathf.RoundToInt(Mathf.Lerp(40f, 255f, strength)));
                _vibrator.Call("vibrate", fx);
            }
            catch (System.Exception) { Handheld.Vibrate(); }   // old Android; also what makes Unity add the VIBRATE permission
#endif
        }
    }
}
