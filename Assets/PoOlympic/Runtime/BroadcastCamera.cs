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
        public float followSharpness = 6f;
        Camera _cam;

        void Awake() => _cam = GetComponent<Camera>();

        void LateUpdate()
        {
            ApplyLetterbox();
            if (target == null) return;
            var p = target.transform.position;
            var focus = new Vector3(p.x, lookHeight, p.z);
            var want = focus + offset;
            float k = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
            transform.position = Vector3.Lerp(transform.position, want, Application.isPlaying ? k : 1f);
            transform.rotation = Quaternion.LookRotation(focus - transform.position, Vector3.up);
        }

        void ApplyLetterbox()
        {
            if (_cam.targetTexture != null) { _cam.rect = new Rect(0, 0, 1, 1); return; }
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
