using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Portrait (9:16) broadcast camera: letterboxes the viewport to the target aspect on any window and smoothly
    /// tracks a MuJoCo body (normally the pelvis) from a side-on track position.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    [DefaultExecutionOrder(200)]
    public class BroadcastCamera : MonoBehaviour
    {
        public MjBody target;
        public Vector2 aspect = new(9, 16);
        public Vector3 offset = new(0.6f, 0.35f, -4.2f);
        public float lookHeight = 0.9f;
        [Tooltip("Added to the target position before framing (e.g. centre of a row of competitors).")]
        public Vector3 focusOffset;
        public float followSharpness = 6f;
        Camera _cam;
        bool _hasViewport;
        Rect _viewport;

        /// <summary>Render into this normalised rect instead of the letterboxed 9:16 one (BroadcastHud: the gap between
        /// its top and bottom docks, so the HUD never covers the game).</summary>
        public void SetViewport(Rect r) { _viewport = r; _hasViewport = true; }
        public void ClearViewport() => _hasViewport = false;

        void Awake()
        {
            _cam = GetComponent<Camera>();
            // The letterboxed viewport leaves bars nobody clears (after a scene load they keep the previous scene's
            // frame): a full-screen camera that renders nothing clears them to black first.
            var bars = new GameObject("LetterboxClear").AddComponent<Camera>();
            bars.transform.SetParent(transform, false);
            bars.clearFlags = CameraClearFlags.SolidColor;
            bars.backgroundColor = Color.black;
            bars.cullingMask = 0;
            bars.depth = _cam.depth - 1;
        }

        void LateUpdate()
        {
            ApplyLetterbox();
            if (target == null) return;
            var p = target.transform.position + focusOffset;
            var focus = new Vector3(p.x, lookHeight, p.z);
            var want = focus + offset;
            float k = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
            transform.position = Vector3.Lerp(transform.position, want, Application.isPlaying ? k : 1f);
            transform.rotation = Quaternion.LookRotation(focus - transform.position, Vector3.up);
        }

        void ApplyLetterbox()
        {
            if (_cam.targetTexture != null) { _cam.rect = new Rect(0, 0, 1, 1); return; }
            if (_hasViewport) { _cam.rect = _viewport; return; }
            float target = aspect.x / aspect.y;
            float window = (float)Screen.width / Mathf.Max(1, Screen.height);
            if (window > target)
            {
                float w = target / window;
                _cam.rect = new Rect((1f - w) * 0.5f, 0f, w, 1f);
            }
            else
            {
                float h = window / target;
                _cam.rect = new Rect(0f, (1f - h) * 0.5f, 1f, h);
            }
        }
    }
}
