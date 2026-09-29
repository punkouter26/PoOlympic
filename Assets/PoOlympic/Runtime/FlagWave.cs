using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Roof flag in the wind: the flag mesh pivots on its pole (build_realism.py sets the pivot there), so a noisy yaw
    /// swing about the pole plus a small droop reads as flapping from the broadcast cameras. Render-only.
    /// </summary>
    public sealed class FlagWave : MonoBehaviour
    {
        [Tooltip("Mean yaw swing (deg) and gust amplitude (deg).")]
        public float swingDeg = 14f;
        public float gustDeg = 10f;
        [Tooltip("Swing frequency (Hz).")]
        public float frequency = 0.9f;
        public float droopDeg = 4f;

        Quaternion rest;
        float seed;

        void Awake()
        {
            rest = transform.localRotation;
            seed = Random.value * 100f;
        }

        void Update()
        {
            float t = Time.time;
            float gust = Mathf.PerlinNoise(seed, t * 0.15f);                     // slow gusts 0..1
            float yaw = Mathf.Sin((t + seed) * 2f * Mathf.PI * frequency) * (swingDeg * 0.5f + gustDeg * gust)
                        + (Mathf.PerlinNoise(seed + 7f, t * 1.7f) - 0.5f) * swingDeg;
            float droop = (1f - gust) * droopDeg;
            transform.localRotation = rest * Quaternion.Euler(0f, yaw, 0f) * Quaternion.AngleAxis(droop, Vector3.forward);
        }
    }
}
