using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Event 14 The Alpine Ramp is simulated as a flat track with gravity tilted by the slope (TrackRaceEvent.Mode.Ramp),
    /// and EventScenes turns the stadium, sun and fill lights by −slope about the ramp's foot so the stadium's ramp lies on
    /// the flat track. This turns the finished camera pose by the same angle about the same axis (through `pivot`, the
    /// focus athlete), after the broadcast rigs have placed it: on screen the stadium is level and the athletes climb.
    /// </summary>
    [DefaultExecutionOrder(32000)]
    public class RampView : MonoBehaviour
    {
        public float slopeDeg = 15f;
        [Tooltip("The turn axis (Unity z = across the lanes) passes through this transform.")]
        public Transform pivot;

        void LateUpdate()
        {
            var p = pivot != null ? pivot.position : Vector3.zero;
            transform.RotateAround(p, Vector3.forward, -slopeDeg);
        }
    }
}
