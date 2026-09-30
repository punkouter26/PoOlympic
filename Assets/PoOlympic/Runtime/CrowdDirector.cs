using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// GFX idea 3 — the crowd follows the heat. Drives the PoOlympic/Crowd shader globals (render only):
    ///   excite  = TensionMeter.Tension (bob + share of fans on their feet)
    ///   cheer   = bursts: the start, a save, the result (long), a world record (longest)
    ///   wave    = a Mexican wave in quiet stretches of a long live heat and while the next heat gets ready
    ///   flash   = camera flashes at the result / a record (ArenaAudio plays the shutter clicks)
    /// </summary>
    [DefaultExecutionOrder(160)]
    public class CrowdDirector : MonoBehaviour
    {
        public MonoBehaviour board;                    // IBroadcastBoard
        public TensionMeter tension;
        public float waveAfterQuietSeconds = 12f, waveSeconds = 9f;

        static readonly int Excite = Shader.PropertyToID("_PoCrowdExcite"), Cheer = Shader.PropertyToID("_PoCrowdCheer"),
                            Wave = Shader.PropertyToID("_PoCrowdWave"), Flash = Shader.PropertyToID("_PoCrowdFlash");
        IBroadcastBoard B => board as IBroadcastBoard;
        BoardPhase _phase = BoardPhase.Ready;
        float _excite, _cheer, _cheerUntil, _flashUntil, _quiet, _waveUntil, _wave;

        public float FlashLevel { get; private set; }

        void OnEnable()
        {
            if (tension != null) { tension.Save += OnSave; tension.Fall += OnFall; }
        }

        void OnDisable()
        {
            if (tension != null) { tension.Save -= OnSave; tension.Fall -= OnFall; }
            Shader.SetGlobalFloat(Excite, 0f); Shader.SetGlobalFloat(Cheer, 0f); Shader.SetGlobalFloat(Wave, 0f); Shader.SetGlobalFloat(Flash, 0f);
        }

        void OnSave(PolicyRunner r) => Burst(2.5f);
        void OnFall(PolicyRunner r) => Burst(1.2f);

        /// <summary>A cheer (and optionally flashes) for the next `seconds`.</summary>
        public void Burst(float seconds, float flashSeconds = 0f)
        {
            _cheerUntil = Mathf.Max(_cheerUntil, Time.unscaledTime + seconds);
            _flashUntil = Mathf.Max(_flashUntil, Time.unscaledTime + flashSeconds);
        }

        /// <summary>BroadcastHud / ArenaAudio: a new world record.</summary>
        public void Record() => Burst(9f, 7f);

        void Update()
        {
            float now = Time.unscaledTime, dt = Time.unscaledDeltaTime;
            var b = B;
            float t = tension != null ? tension.Tension : 0.25f;
            if (b != null && b.BoardState != _phase)
            {
                if (b.BoardState == BoardPhase.Live) Burst(2f);
                if (b.BoardState == BoardPhase.Result) Burst(6f, 4.5f);
                _phase = b.BoardState;
                _quiet = 0f;
            }
            // a wave when nothing happens for a while (live and low tension), and between heats
            bool calm = b == null || b.BoardState == BoardPhase.Ready || t < 0.35f;
            _quiet = calm ? _quiet + dt : 0f;
            if (_quiet > waveAfterQuietSeconds && now > _waveUntil + waveSeconds) { _waveUntil = now + waveSeconds; _quiet = 0f; }

            float k = 1f - Mathf.Exp(-4f * dt);
            _excite = Mathf.Lerp(_excite, t, k);
            _cheer = Mathf.Lerp(_cheer, now < _cheerUntil ? 1f : 0f, 1f - Mathf.Exp(-(now < _cheerUntil ? 8f : 1.5f) * dt));
            _wave = Mathf.MoveTowards(_wave, now < _waveUntil ? 1f : 0f, dt * 0.8f);
            FlashLevel = now < _flashUntil ? Mathf.Clamp01((_flashUntil - now) / 2f) : 0f;
            Shader.SetGlobalFloat(Excite, _excite);
            Shader.SetGlobalFloat(Cheer, _cheer);
            Shader.SetGlobalFloat(Wave, _wave);
            Shader.SetGlobalFloat(Flash, FlashLevel);
        }
    }
}
