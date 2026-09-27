using UnityEngine;

namespace PoOlympic
{
    /// <summary>Records which generated MJCF file (and exact bytes) an imported MuJoCo hierarchy came from.</summary>
    public class ModelProvenance : MonoBehaviour
    {
        public string sourceFile;
        public string sha256;
        public string importedUtc;
    }
}
