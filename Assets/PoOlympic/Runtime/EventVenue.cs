using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// One of the 30 PoOlympics events placed in the stadium: catalogue data (events_catalog.json) plus the 8 competitor
    /// spots (Stadium.glb anchors E##_L0..E##_L7, from SourceArt/Stadium/build_venues.py). Render-only — the event's
    /// physics lives in its own event scene / MJCF.
    /// </summary>
    public class EventVenue : MonoBehaviour
    {
        public int number;
        public string eventName;
        public int phase;
        public string skill;
        public string brain;
        [TextArea] public string rules;
        [Tooltip("Scene that runs this event (empty = not built yet).")]
        public string eventScene;
        public Transform[] lanes = new Transform[8];

        public bool Playable => !string.IsNullOrEmpty(eventScene);

        public Vector3 Centre
        {
            get
            {
                var c = Vector3.zero;
                int n = 0;
                foreach (var l in lanes)
                    if (l != null) { c += l.position; n++; }
                return n > 0 ? c / n : transform.position;
            }
        }

        void OnDrawGizmos()
        {
            Gizmos.color = Color.HSVToRGB((phase - 1) / 6f, 0.8f, 1f);
            for (int i = 0; i < lanes.Length; i++)
            {
                if (lanes[i] == null) continue;
                Gizmos.DrawWireSphere(lanes[i].position + Vector3.up * 0.9f, 0.35f);
                Gizmos.DrawLine(lanes[i].position, lanes[i].position + Vector3.up * 1.8f);
            }
        }
    }
}
