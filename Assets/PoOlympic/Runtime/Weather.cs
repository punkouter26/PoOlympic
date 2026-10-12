using System;
using Mujoco;
using UnityEngine;
using UnityEngine.Networking;

namespace PoOlympic
{
    /// <summary>
    /// Real-weather meets (opt-in: GameSettings.RealWeather, off = the nominal physics of the parity gates and odds).
    /// Today's host city (one per day of the year) is asked once per session at Open-Meteo (free, no key, no player
    /// data in the request); offline or refused = no weather.
    ///   wind   drag on every athlete's torso, each physics step (PolicyRunner → mjData.xfrc_applied):
    ///          F = ½ ρ CdA w², CdA = DragArea × the body's length scale², w gusting between the mean and the gust speed,
    ///          mean capped at MaxWind. North = the running direction (+x), so a north wind is a headwind.
    ///   rain   from WetRainMm of rain per hour: sliding friction of every geom × WetGrip (inside the ±20 % friction
    ///          range the brains were trained on), set when a scene starts.
    /// CPU check (dash, mixed line-up, wind 6 / 9 / 12 m/s from ahead, behind and the side, dry and wet): 1 fall in 576
    /// runs — the brains lean into it; the weather shifts times by a few tenths, it rarely floors anyone.
    /// </summary>
    public static unsafe class Weather
    {
        public const float MaxWind = 12f, MaxGustFactor = 1.4f, GustPeriod = 7f, DragArea = 0.6f, AirDensity = 1.2f;
        public const float WetGrip = 0.85f, WetRainMm = 1f;

        static readonly (string city, float lat, float lon)[] Hosts =
        {
            ("Athens", 37.98f, 23.73f), ("Paris", 48.85f, 2.35f), ("London", 51.51f, -0.13f), ("Tokyo", 35.68f, 139.69f),
            ("Rio de Janeiro", -22.91f, -43.17f), ("Sydney", -33.87f, 151.21f), ("Los Angeles", 34.05f, -118.24f),
            ("Oslo", 59.91f, 10.75f), ("Beijing", 39.90f, 116.40f), ("Barcelona", 41.39f, 2.17f),
            ("Reykjavik", 64.15f, -21.94f), ("Wellington", -41.29f, 174.78f),
        };

        [Serializable] class Current { public float temperature_2m, precipitation, wind_speed_10m, wind_direction_10m, wind_gusts_10m; }
        [Serializable] class Reply { public Current current; }

        public static bool Ready { get; private set; }
        public static bool Active => Ready && GameSettings.RealWeather;
        public static string City { get; private set; } = "";
        public static float WindMps { get; private set; }
        public static float GustMps { get; private set; }
        /// <summary>Where the wind comes from, degrees clockwise from north (meteorological).</summary>
        public static float WindFromDeg { get; private set; }
        public static float RainMm { get; private set; }
        public static float TempC { get; private set; }
        public static bool Wet => RainMm >= WetRainMm * 0.1f;

        static bool _asked;
        static MjScene _wetScene;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() { Ready = false; _asked = false; _wetScene = null; City = ""; }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot() { GameSettings.Changed += Fetch; Fetch(); }

        static void Fetch()
        {
            if (_asked || !GameSettings.RealWeather) return;
            _asked = true;
            var h = Hosts[DateTime.UtcNow.DayOfYear % Hosts.Length];
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var req = UnityWebRequest.Get("https://api.open-meteo.com/v1/forecast?latitude=" + h.lat.ToString(inv) + "&longitude=" + h.lon.ToString(inv) +
                                          "&current=temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,wind_gusts_10m&wind_speed_unit=ms");
            req.timeout = 8;
            req.SendWebRequest().completed += _ =>
            {
                try
                {
                    var c = req.result == UnityWebRequest.Result.Success ? JsonUtility.FromJson<Reply>(req.downloadHandler.text)?.current : null;
                    if (c == null) { Debug.LogWarning($"[Weather] no weather for {h.city}: {req.error} — calm and dry."); return; }
                    Set(h.city, c.wind_speed_10m, c.wind_gusts_10m, c.wind_direction_10m, c.precipitation, c.temperature_2m);
                }
                finally { req.Dispose(); }
            };
        }

        /// <summary>The weather of the session (the Open-Meteo reply; tests set it directly).</summary>
        public static void Set(string city, float windMps, float gustMps, float fromDeg, float rainMm, float tempC)
        {
            City = city; WindMps = Mathf.Max(0f, windMps); GustMps = Mathf.Max(WindMps, gustMps);
            WindFromDeg = fromDeg; RainMm = Mathf.Max(0f, rainMm); TempC = tempC;
            Ready = true;
        }

        /// <summary>Wind speed (m/s) at simulation time t: the capped mean, swelling to the gust speed every GustPeriod.</summary>
        public static double SpeedAt(double t)
        {
            double mean = Math.Min(WindMps, MaxWind);
            double gust = mean > 0.1 ? Math.Clamp(GustMps / WindMps, 1.0, MaxGustFactor) : 1.0;
            return mean * (1.0 + (gust - 1.0) * 0.5 * (1.0 + Math.Sin(2.0 * Math.PI * t / GustPeriod)));
        }

        /// <summary>Drag (N, MuJoCo world x / y) on a body whose frontal area is areaScale × MATT's.</summary>
        public static (double x, double y) WindForce(double t, double areaScale)
        {
            double w = SpeedAt(t), f = 0.5 * AirDensity * DragArea * areaScale * w * w;
            double to = (WindFromDeg + 180.0) * Math.PI / 180.0;      // blows towards; north = +x, east = −y (y is left)
            return (f * Math.Cos(to), -f * Math.Sin(to));
        }

        /// <summary>Rain: scale the sliding friction of every geom of this scene's model, once per scene.</summary>
        public static void WetTrack(MujocoLib.mjModel_* m)
        {
            if (!Active || !Wet || !MjScene.InstanceExists || MjScene.Instance == _wetScene) return;
            _wetScene = MjScene.Instance;
            double k = Mathf.Lerp(1f, WetGrip, Mathf.Clamp01(RainMm / WetRainMm));
            for (int g = 0; g < (int)m->ngeom; g++) m->geom_friction[3 * g] *= k;
        }

        /// <summary>One ticker line: "Oslo 9°C · 7 m/s headwind, gusts 11 · rain, wet track".</summary>
        public static string Headline
        {
            get
            {
                double to = (WindFromDeg + 180.0) * Math.PI / 180.0, along = Math.Cos(to);
                string dir = WindMps < 1f ? "calm" : $"{WindMps:0.#} m/s " + (along > 0.5 ? "tailwind" : along < -0.5 ? "headwind" : "crosswind") +
                             (GustMps > WindMps + 1f ? $", gusts {GustMps:0}" : "");
                return $"{City} {TempC:0}°C · {dir}" + (Wet ? " · rain, wet track" : "");
            }
        }
    }
}
