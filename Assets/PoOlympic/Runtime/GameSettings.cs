using System;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Player options (PlayerPrefs, per device), edited in the settings sheet of the HudAnchors frame (menu and every
    /// event). Defaults keep the earlier rulings: no camera shake (user, 2026-09-30: "fast movements hurt my eyes"),
    /// no betting slip (user, 2026-09-29: "just play"), real-time heats. Nothing here touches ctrl or the physics
    /// step size: SimSpeed is Time.timeScale, so a slowed heat is the same simulation watched slower.
    /// </summary>
    public static class GameSettings
    {
        /// <summary>Raised after any option changes (listeners re-read what they use).</summary>
        public static event Action Changed;

        // read every frame by the audio mix and the camera: cached, PlayerPrefs only on first read and on change
        static readonly System.Collections.Generic.Dictionary<string, float> Cache = new();
        static float F(string key, float def) =>
            Cache.TryGetValue(key, out var v) ? v : Cache[key] = PlayerPrefs.GetFloat("poolympic.set." + key, def);
        static bool B(string key, bool def) => F(key, def ? 1f : 0f) > 0.5f;
        static void Set(string key, float v) { Cache[key] = v; PlayerPrefs.SetFloat("poolympic.set." + key, v); Changed?.Invoke(); }
        static void Set(string key, bool v) => Set(key, v ? 1f : 0f);

        // ---- audio: multipliers on ArenaAudio's own mix (the mixer snapshots keep doing the phase ducking)
        public static float VolMaster { get => F("vol", 1f); set => Set("vol", Mathf.Clamp01(value)); }
        public static float VolCrowd { get => F("volCrowd", 1f); set => Set("volCrowd", Mathf.Clamp01(value)); }
        public static float VolSfx { get => F("volSfx", 1f); set => Set("volSfx", Mathf.Clamp01(value)); }
        public static float VolAnnouncer { get => F("volPa", 1f); set => Set("volPa", Mathf.Clamp01(value)); }

        // ---- display
        /// <summary>0 = tripod (default) … 1 = full impact shake (BroadcastDirector → Cinemachine impulse gain).</summary>
        public static float CameraShake { get => F("shake", 0f); set => Set("shake", Mathf.Clamp01(value)); }
        /// <summary>Close-ups ease in a little as the athlete on camera gets into trouble.</summary>
        public static bool TensionZoom { get => B("tensionZoom", true); set => Set("tensionZoom", value); }
        /// <summary>Depth of field on close-ups, the bloom pulse on a record and the colour-fringe flash on a big slam
        /// (off by default on phones).</summary>
        public static bool FocusBlur { get => B("focusBlur", !Application.isMobilePlatform); set => Set("focusBlur", value); }
        /// <summary>Scuff marks where a body or a cube hits the ground.</summary>
        public static bool ImpactMarks { get => B("marks", true); set => Set("marks", value); }

        /// <summary>Draw the MuJoCo collision shapes and contact forces over the athletes (ColliderXray).</summary>
        public static bool ColliderView { get => B("colliders", false); set => Set("colliders", value); }

        /// <summary>Phone vibration on body slams and cube hits, scaled by the contact force.</summary>
        public static bool Haptics { get => B("haptics", true); set => Set("haptics", value); }

        // ---- game
        /// <summary>Today's real weather of the host city (Open-Meteo, one request per session): wind pushes the
        /// athletes, rain makes the track slippery. Off = the nominal physics of the parity gates and the odds.</summary>
        public static bool RealWeather { get => B("weather", false); set => Set("weather", value); }
        /// <summary>Play-by-play lines rewritten by Claude (needs the player's own API key, see AiCommentator).</summary>
        public static bool AiCommentary { get => B("aiPbp", false); set => Set("aiPbp", value); }
        public static bool OfferBets { get => B("bets", false); set => Set("bets", value); }
        /// <summary>Per-heat action log (PolicyRunner → persistentDataPath/actions): every control tick of every lane.</summary>
        public static bool LogActions { get => B("logActions", Application.isEditor); set => Set("logActions", value); }

        public static readonly float[] Speeds = { 1f, 0.5f, 0.25f, 2f };
        static float _speed = 1f;
        /// <summary>Viewing speed of the simulation (this session only): 1, 0.5, 0.25 or 2.</summary>
        public static float SimSpeed
        {
            get => _speed;
            set
            {
                _speed = Mathf.Clamp(value, 0.1f, 4f);
                if (Time.timeScale > 0f) Time.timeScale = _speed;      // not during a hit-stop: ImpactFx restores to SimSpeed
                Changed?.Invoke();
            }
        }

        /// <summary>The next speed of <see cref="Speeds"/> (menu sheet: one item that cycles).</summary>
        public static void NextSpeed() => SimSpeed = Speeds[(Array.IndexOf(Speeds, _speed) + 1) % Speeds.Length];

        public static string SpeedLabel => _speed == 1f ? "1×" : $"{_speed:0.##}×";

        /// <summary>Wipe the save: wallet, every world record, the demo season. Options stay.</summary>
        public static void ResetSave(System.Collections.Generic.IEnumerable<int> eventNumbers)
        {
            Wallet.Coins = Wallet.StartCoins;
            foreach (var ev in eventNumbers) Records.Clear(ev);
            DemoSeason.Reset();
            Changed?.Invoke();
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics()
        {
            Changed = null;
            Cache.Clear();
            _speed = 1f;
            Time.timeScale = 1f;
        }
    }
}
