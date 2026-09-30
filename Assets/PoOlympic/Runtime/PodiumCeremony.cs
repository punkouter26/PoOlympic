using System.Collections.Generic;
using System.Linq;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// GFX idea 6 — medal ceremony on the stadium podium (Stadium.glb: Podium, Podium_Step_1-3, Podium_Flag_1-3,
    /// Podium_FlagTop_1-3, PodiumCam, PodiumLook). A few seconds into the result the top three are frozen as "photo
    /// statues" (SkinnedMeshRenderer.BakeMesh of their pose at that moment) on the steps, the gold / silver / bronze
    /// flags go up, confetti, fanfare, camera flashes, and BroadcastDirector cuts to the podium camera for `hold`
    /// seconds before going back to the winner. Everything is cleared when the next heat starts. Render / audio only:
    /// the physics athletes stay where they are. The statues are pre-authored slots (statueSlots: pivot › pose › parts
    /// with MeshFilter + MeshRenderer, built by StadiumShowcase); their meshes are allocated once and re-baked each
    /// ceremony (no Instantiate / Destroy at runtime).
    /// </summary>
    [DefaultExecutionOrder(175)]
    public class PodiumCeremony : MonoBehaviour
    {
        public MonoBehaviour board;                    // IBroadcastBoard
        public Vector3 raceForward = Vector3.right;
        public Transform[] steps = new Transform[3];    // 1st, 2nd, 3rd
        public Transform[] flags = new Transform[3];
        public Transform[] flagTops = new Transform[3];
        public Transform podiumCam, podiumLook;
        public ParticleSystem confetti;
        public ArenaAudio arenaAudio;
        public CrowdDirector crowd;
        public float delay = 2.5f, hold = 5.5f, flagSeconds = 4f;
        public Color[] flagColors = { new(1f, 0.8f, 0.25f), new(0.82f, 0.85f, 0.9f), new(0.8f, 0.5f, 0.25f) };
        [Tooltip("One per step: pivot (child 0 = pose root, whose children are the parts).")]
        public Transform[] statueSlots = new Transform[3];

        /// <summary>BroadcastDirector: show the podium shot now.</summary>
        public bool ShowPodium => _state == State.Ceremony;

        enum State { Idle, Waiting, Ceremony, Done }
        State _state;
        float _clock;
        IBroadcastBoard B => board as IBroadcastBoard;
        Vector3[] _flagLow;
        MaterialPropertyBlock _mpb;

        void Start()
        {
            _flagLow = flags.Select(f => f != null ? f.position : Vector3.zero).ToArray();
            _mpb = new MaterialPropertyBlock();
            foreach (var slot in statueSlots)
            {
                if (slot == null) continue;
                foreach (var mf in slot.GetComponentsInChildren<MeshFilter>(true)) mf.sharedMesh = new Mesh { name = "statue_" + mf.name };
                slot.gameObject.SetActive(false);
            }
            for (int i = 0; i < flags.Length; i++)
            {
                if (flags[i] == null) continue;
                var r = flags[i].GetComponentInChildren<Renderer>(true);     // flags are parked inactive
                if (r != null) { r.GetPropertyBlock(_mpb); _mpb.SetColor("baseColorFactor", flagColors[i]); _mpb.SetColor("_BaseColor", flagColors[i]); r.SetPropertyBlock(_mpb); }
                flags[i].gameObject.SetActive(false);
            }
        }

        void Update()
        {
            var b = B;
            if (b == null) return;
            if (b.BoardState != BoardPhase.Result)
            {
                if (_state != State.Idle) Clear();
                return;
            }
            _clock += Time.unscaledDeltaTime;
            switch (_state)
            {
                case State.Idle: _state = State.Waiting; _clock = 0f; break;
                case State.Waiting when _clock >= delay: Begin(b); break;
                case State.Ceremony:
                    float u = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(_clock / flagSeconds));
                    for (int i = 0; i < flags.Length; i++)
                        if (flags[i] != null && flagTops[i] != null)
                            flags[i].position = Vector3.Lerp(_flagLow[i], flagTops[i].position, u * (i == 0 ? 1f : 0.93f));
                    if (_clock >= hold) _state = State.Done;
                    break;
            }
        }

        void Begin(IBroadcastBoard b)
        {
            _state = State.Ceremony;
            _clock = 0f;
            var top = b.Rows.Where(r => r.place >= 1 && r.place <= 3).OrderBy(r => r.place).ToList();
            foreach (var r in top) Statue(r.runner, r.place - 1);
            for (int i = 0; i < flags.Length; i++) if (flags[i] != null) { flags[i].position = _flagLow[i]; flags[i].gameObject.SetActive(i < Mathf.Max(1, top.Count)); }
            if (confetti != null) { confetti.Clear(); confetti.Play(); }
            if (arenaAudio != null) arenaAudio.Fanfare();
            if (crowd != null) crowd.Burst(hold, hold * 0.8f);
        }

        void Clear()
        {
            foreach (var slot in statueSlots) if (slot != null) slot.gameObject.SetActive(false);
            for (int i = 0; i < flags.Length; i++) if (flags[i] != null) { flags[i].position = _flagLow[i]; flags[i].gameObject.SetActive(false); }
            _state = State.Idle;
            _clock = 0f;
        }

        /// <summary>Freeze the athlete's current pose on step `k` (0 = gold), turned to face the podium camera.</summary>
        void Statue(PolicyRunner r, int k)
        {
            if (r == null || k < 0 || k >= steps.Length || steps[k] == null || k >= statueSlots.Length || statueSlots[k] == null) return;
            var slot = statueSlots[k];
            if (slot.childCount == 0) return;
            var pose = slot.GetChild(0);
            var parts = pose.GetComponentsInChildren<MeshFilter>(true);
            var skins = r.GetComponentsInChildren<SkinnedMeshRenderer>(false).Where(s => s.enabled && s.gameObject.activeInHierarchy).ToList();
            if (skins.Count == 0 || parts.Length == 0) return;
            var pelvis = FindObjectsByType<MjBody>().FirstOrDefault(x => x.name == r.athletePrefix + "pelvis");
            var hip = pelvis != null ? pelvis.transform.position : skins[0].bounds.center;
            slot.gameObject.SetActive(true);
            slot.SetPositionAndRotation(hip, Quaternion.identity);
            pose.SetLocalPositionAndRotation(Vector3.zero, Quaternion.identity);
            float minY = float.MaxValue;
            for (int i = 0; i < parts.Length; i++)
            {
                var mf = parts[i];
                bool on = i < skins.Count;
                mf.gameObject.SetActive(on);
                if (!on) continue;
                var s = skins[i];
                s.BakeMesh(mf.sharedMesh, true);
                mf.transform.SetPositionAndRotation(s.transform.position, s.transform.rotation);
                var mr = mf.GetComponent<MeshRenderer>();
                mr.sharedMaterials = s.sharedMaterials;
                minY = Mathf.Min(minY, mr.bounds.min.y);
            }
            // pivot: under the pelvis at the lowest point of the pose; turn from the race direction to the podium camera
            pose.localPosition = new Vector3(0f, hip.y - minY, 0f);
            var from = new Vector3(raceForward.x, 0, raceForward.z).normalized;
            var toCam = podiumCam != null ? podiumCam.position - steps[k].position : -from;
            toCam.y = 0;
            slot.SetPositionAndRotation(steps[k].position, Quaternion.Euler(0f, Vector3.SignedAngle(from, toCam.normalized, Vector3.up), 0f));
        }
    }
}
