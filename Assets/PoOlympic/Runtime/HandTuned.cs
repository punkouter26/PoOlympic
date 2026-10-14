using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Marker: this object was placed by hand in a built event scene. When the editor builder rebuilds the scene
    /// (EventScenes "Build Event …"), the object at the same hierarchy path gets this local position / rotation / scale
    /// back (Editor/HandTunedSave.cs). Add it in the Inspector after moving a camera, light or prop.
    /// </summary>
    [DisallowMultipleComponent]
    public class HandTuned : MonoBehaviour { }
}
