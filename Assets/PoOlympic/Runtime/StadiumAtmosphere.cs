using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Render-only stadium look (sky, fog, ambient) applied to the scene's RenderSettings. Lives on the
    /// StadiumAtmosphere prefab (with the post-processing Volume, reflection probe and floodlights as children) so the
    /// values can be tuned in the Inspector instead of in code. Never touches physics.
    /// </summary>
    [ExecuteAlways]
    public sealed class StadiumAtmosphere : MonoBehaviour
    {
        public Material skybox;
        [Range(0f, 2f)] public float ambientIntensity = 1.0f;
        public bool fog = true;
        public Color fogColor = new Color(0.72f, 0.78f, 0.86f);
        [Tooltip("Linear fog start/end (m): the skyline ring sits 420-700 m out, the park 95-355 m.")]
        public float fogStart = 180f;
        public float fogEnd = 950f;

        void OnEnable() => Apply();
        void OnValidate() => Apply();

        public void Apply()
        {
            if (skybox != null) RenderSettings.skybox = skybox;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Skybox;
            RenderSettings.ambientIntensity = ambientIntensity;
            RenderSettings.fog = fog;
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogColor = fogColor;
            RenderSettings.fogStartDistance = fogStart;
            RenderSettings.fogEndDistance = fogEnd;
            DynamicGI.UpdateEnvironment();
        }
    }
}
