using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>An event whose lane list can drop the athletes LaneLineup switched off (roster scenes).</summary>
    public interface ILaneRoster
    {
        void DropInactiveLanes();
    }

    /// <summary>
    /// Roster event scenes (training/tools/compose_mixed.py &lt;scene&gt; roster) hold EVERY roster body in every lane:
    /// MATT as L&lt;k&gt;_, the zombie as Z&lt;k&gt;_, same lane collision bits. Before MuJoCo compiles the scene (MjScene.Start),
    /// this keeps the body picked in the main menu (MeetLineup.Athletes; MATT when that body is missing) and switches
    /// off the others — runner, visual, physics bodies, their actuators and contact excludes — so the running model is
    /// compose(scene, lineup). Then each event drops the switched-off athletes and the camera follows the kept pelvis.
    /// </summary>
    [DefaultExecutionOrder(-10000)]
    public sealed class LaneLineup : MonoBehaviour
    {
        [Serializable]
        public class Entry
        {
            public int lane;
            public string body;          // lower case (contract body)
            public PolicyRunner runner;
            public GameObject pelvis;    // root MjBody of this athlete in the physics scene
        }

        public List<Entry> entries = new();
        public GameObject physicsRoot;
        public MjCubePool pool;
        public BroadcastCamera broadcastCamera;
        public int focusLane = 3;

        /// <summary>Body kept in each lane (after Awake).</summary>
        public string[] Kept { get; private set; } = new string[0];

        void Awake()
        {
            int lanes = entries.Count == 0 ? 0 : entries.Max(e => e.lane) + 1;
            Kept = new string[lanes];
            for (int k = 0; k < lanes; k++)
            {
                string want = k < MeetLineup.Athletes.Length ? MeetLineup.Athletes[k].ToLowerInvariant() : "matt";
                if (!entries.Any(e => e.lane == k && e.body == want)) want = "matt";
                Kept[k] = want;
                foreach (var e in entries.Where(e => e.lane == k && e.body != want))
                {
                    e.runner.gameObject.SetActive(false);
                    e.pelvis.SetActive(false);
                }
            }
            // actuators / excludes live outside the athlete trees: switch off those that reference switched-off parts
            foreach (var a in physicsRoot.GetComponentsInChildren<MjActuator>(true))
                if (a.Joint != null && !a.Joint.gameObject.activeInHierarchy) a.gameObject.SetActive(false);
            foreach (var x in physicsRoot.GetComponentsInChildren<MjExclude>(true))
                if ((x.Body1 != null && !x.Body1.gameObject.activeInHierarchy) || (x.Body2 != null && !x.Body2.gameObject.activeInHierarchy))
                    x.gameObject.SetActive(false);

            foreach (var ev in FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None).OfType<ILaneRoster>())
                ev.DropInactiveLanes();

            if (pool != null && (pool.runner == null || !pool.runner.gameObject.activeInHierarchy))
                pool.runner = entries.First(e => e.body == Kept[e.lane]).runner;
            if (broadcastCamera != null)
            {
                var focus = entries.FirstOrDefault(e => e.lane == focusLane && e.body == Kept[focusLane]);
                if (focus != null) broadcastCamera.target = focus.pelvis.GetComponent<MjBody>();
            }
            Debug.Log($"[LaneLineup] {string.Join(" ", Kept.Select((b, k) => $"{char.ToUpperInvariant(b[0])}{k + 1}"))}");
        }
    }
}
