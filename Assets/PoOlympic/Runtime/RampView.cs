using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Event 14 The Alpine Ramp is simulated as a flat track with gravity tilted by the slope (TrackRaceEvent.Mode.Ramp),
    /// and EventScenes turns the stadium, sun and fill lights by −slope about the ramp's foot so the stadium's ramp lies on
    /// the flat track. After the broadcast rigs have placed the camera, this keeps its position (above the track = above the
    /// ramp) and where it looks, and turns its up direction by −slope like the stadium: on screen the stadium is level and
    /// the athletes climb.
    /// </summary>
    [DefaultExecutionOrder(32000)]
    public class RampView : MonoBehaviour
    {
        public float slopeDeg = 15f;

        void LateUpdate()
        {
            var up = Quaternion.AngleAxis(-slopeDeg, Vector3.forward) * Vector3.up;
            transform.rotation = Quaternion.LookRotation(transform.forward, up);
        }
    }
}
