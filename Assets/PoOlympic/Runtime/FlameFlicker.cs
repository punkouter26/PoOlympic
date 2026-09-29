using UnityEngine;

namespace PoOlympic
{
    /// <summary>Olympic cauldron flame: noisy height / twist and a flickering point light. Render-only.</summary>
    public sealed class FlameFlicker : MonoBehaviour
    {
        public Light glow;
        public float heightJitter = 0.18f;
        public float twistDegPerSec = 25f;
        public float speed = 3.5f;
        public float glowIntensity = 8f;

        Vector3 rest;

        void Awake() => rest = transform.localScale;

        void Update()
        {
            float t = Time.time * speed;
            float n = Mathf.PerlinNoise(t, 0.37f);
            float w = Mathf.PerlinNoise(0.71f, t * 1.3f);
            transform.localScale = new Vector3(rest.x * (1f + 0.08f * (w - 0.5f)), rest.y * (1f + heightJitter * (n - 0.5f) * 2f),
                                               rest.z * (1f + 0.08f * (0.5f - w)));
            transform.Rotate(0f, twistDegPerSec * Time.deltaTime, 0f, Space.Self);
            if (glow != null) glow.intensity = glowIntensity * (0.8f + 0.4f * n);
        }
    }
}
