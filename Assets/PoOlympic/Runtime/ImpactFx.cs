using System.Collections;
using System.Collections.Generic;
using Unity.Cinemachine;
using UnityEngine;
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
        public CinemachineImpulseSource impulse;

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

        [Header("Hit-stop")]
        public bool hitStop = true;
        public float hitStopSeconds = 0.07f;
        public float hitStopCooldown = 2.5f;
        [Tooltip("Freeze frames while a heat is live. Off (user, 2026-09-29: \"no slo motion during the race\"): the " +
                 "live heat runs at full speed; shake and effects still play.")]
        public bool hitStopWhileLive;

        public int Emitted { get; private set; }

        int _dust, _slam, _wave, _spark;
        float _lastStop = -99f;
        bool _stopping;
        float _stopBefore = 1f, _stopUntil;
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
            if (_stopping) { Time.timeScale = 1f; _stopping = false; }
        }

        void OnImpact(ImpactEvent e)
        {
            switch (e.kind)
            {
                case ImpactEvent.Kind.FootStrike:
                    if (e.speedMps < footDustMinSpeed || footDust == null || footDust.Length == 0) return;
                    if (_lastPuff.TryGetValue(e.runner, out var t) && Time.time - t < 0.2f) return;
                    _lastPuff[e.runner] = Time.time;
                    Burst(footDust, ref _dust, e.position + Vector3.up * 0.02f, Mathf.Clamp01((e.speedMps - footDustMinSpeed) / 3f));
                    break;
                case ImpactEvent.Kind.BodySlam:
                    float need = slamMinWeights * (e.runner != null && e.runner.crawlSteering ? crawlerSlamFactor : 1f);
                    if (e.weightRatio < need) return;
                    float s = Mathf.Clamp01((e.weightRatio - need) / (bigSlamWeights * 2f));
                    Burst(slamDust, ref _slam, e.position + Vector3.up * 0.03f, s);
                    Burst(shockwaves, ref _wave, new Vector3(e.position.x, e.position.y + 0.02f, e.position.z), s);
                    if (e.weightRatio >= bigSlamWeights * (need / slamMinWeights)) Big(e, 0.25f + 0.5f * s);
                    break;
                case ImpactEvent.Kind.CubeHit:
                    if (e.weightRatio < cubeMinWeights) return;
                    float c = Mathf.Clamp01(e.weightRatio / (cubeHitStopWeights * 3f));
                    Sparks(e.position, e.normal, c);
                    if (e.weightRatio >= cubeHitStopWeights) Big(e, 0.3f + 0.5f * c);
                    break;
            }
        }

        void Burst(ParticleSystem[] pool, ref int next, Vector3 at, float strength)
        {
            if (pool == null || pool.Length == 0) return;
            var ps = pool[next];
            next = (next + 1) % pool.Length;
            if (ps == null) return;
            ps.transform.position = at;
            ps.transform.localScale = Vector3.one * Mathf.Lerp(0.8f, 1.6f, strength);
            ps.Play(true);
            Emitted++;
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
            if (!hitStop || _stopping || Time.unscaledTime - _lastStop < hitStopCooldown) return;
            if (!hitStopWhileLive && HeatLive()) return;                // no slow motion during the race
            _stopping = true;
            _lastStop = Time.unscaledTime;
            _stopBefore = Time.timeScale;
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

        void Update()
        {
            if (!_stopping || Time.realtimeSinceStartup < _stopUntil) return;
            Time.timeScale = _stopBefore > 0f ? _stopBefore : 1f;
            _stopping = false;
        }
    }
}
