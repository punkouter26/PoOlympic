using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace PoOlympic
{
    /// <summary>
    /// AI play-by-play (opt-in: GameSettings.AiCommentary + the player's own Anthropic API key): a plain ticker line of
    /// <see cref="Commentary"/> is sent to Claude with the heat's standings and comes back as one broadcast line, which
    /// replaces the plain one in the ticker. One request at a time, at most one per MinGapSeconds; any failure keeps the
    /// plain line. Read-only: nothing here reaches the simulation.
    /// Key: environment variable ANTHROPIC_API_KEY, or the first line of &lt;persistentDataPath&gt;/anthropic_api_key.txt
    /// (phones). It is never bundled with the app and never logged.
    /// Raw HTTPS (UnityWebRequest) because the Anthropic C# SDK is a NuGet package, not a Unity package.
    /// </summary>
    public static class AiCommentator
    {
        public const string Model = "claude-haiku-5-5";
        public const float MinGapSeconds = 3f;
        public const int MaxChars = 110;
        const string Url = "https://api.anthropic.com/v1/messages";
        const string SystemPrompt =
            "You are the stadium play-by-play commentator of the PoOlympics, a track meet whose athletes are physics-simulated " +
            "bodies driven by neural-network brains: MATT the all-rounder, ZOMBIE the hunched shuffler, GRANDMA frail but steady, " +
            "KIM light and quick. You get the heat so far and one plain call. Rewrite the call as a single vivid broadcast line of " +
            "at most 90 characters. Plain text only: no quotes, no emoji, no line breaks. Use only the facts given - never invent " +
            "names, results or numbers.";

        [Serializable] class Block { public string type; public string text; }
        [Serializable] class Reply { public Block[] content; public string stop_reason; }

        static string _key;
        static bool _busy, _off;
        static float _last = -99f;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() { _key = null; _busy = _off = false; _last = -99f; }

        static string Key
        {
            get
            {
                if (_key != null) return _key;
                _key = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY") ?? "";
                try
                {
                    var file = Path.Combine(Application.persistentDataPath, "anthropic_api_key.txt");
                    if (_key.Length == 0 && File.Exists(file)) _key = File.ReadAllText(file);
                }
                catch (IOException) { }
                return _key = _key.Trim();
            }
        }

        public static bool HasKey => Key.Length > 0;
        public static bool Enabled => Application.isPlaying && GameSettings.AiCommentary && !_off && HasKey;

        /// <summary>Ask for a broadcast version of `line`; when it arrives it replaces `line` in `lines` (if still there).</summary>
        public static void Colour(string line, Func<string> context, List<string> lines)
        {
            if (!Enabled || _busy || Time.unscaledTime - _last < MinGapSeconds) return;
            _busy = true;
            _last = Time.unscaledTime;
            var req = new UnityWebRequest(Url, UnityWebRequest.kHttpVerbPOST)
            {
                uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(Body(line, context?.Invoke() ?? ""))),
                downloadHandler = new DownloadHandlerBuffer(),
                timeout = 10,
            };
            req.SetRequestHeader("content-type", "application/json");
            req.SetRequestHeader("anthropic-version", "2023-06-01");
            req.SetRequestHeader("x-api-key", Key);
            req.SendWebRequest().completed += _ =>
            {
                try
                {
                    if (req.result != UnityWebRequest.Result.Success)
                    {
                        // a bad key or a model this key cannot use will not fix itself: stop for the session
                        if (req.responseCode is 400 or 401 or 403 or 404) _off = true;
                        Debug.LogWarning($"[AiCommentator] HTTP {req.responseCode} {req.error}" + (_off ? " — off for this session." : ""));
                        return;
                    }
                    string text = Parse(req.downloadHandler.text);
                    int i = lines.IndexOf(line);
                    if (text.Length > 0 && i >= 0) lines[i] = text;
                }
                finally { _busy = false; req.Dispose(); }
            };
        }

        /// <summary>The Messages API request (Haiku: thinking off + low effort = a fast one-liner).</summary>
        public static string Body(string line, string context) =>
            "{\"model\":\"" + Model + "\",\"max_tokens\":120,\"thinking\":{\"type\":\"disabled\"},\"output_config\":{\"effort\":\"low\"}," +
            "\"system\":\"" + Esc(SystemPrompt) + "\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
            Esc("Heat so far:\n" + context + "\n\nPlain call: " + line) + "\"}]}";

        /// <summary>The first text block of a reply as one ticker line; empty for a refusal or a reply without text.</summary>
        public static string Parse(string json)
        {
            Reply r;
            try { r = JsonUtility.FromJson<Reply>(json); } catch (ArgumentException) { return ""; }
            if (r?.content == null || r.stop_reason == "refusal") return "";
            foreach (var b in r.content)
            {
                if (b.type != "text" || string.IsNullOrWhiteSpace(b.text)) continue;
                var s = b.text.Replace('\n', ' ').Replace('\r', ' ').Trim().Trim('"');
                return s.Length > MaxChars ? s.Substring(0, MaxChars - 1) + "…" : s;
            }
            return "";
        }

        public static string Esc(string s)
        {
            var sb = new StringBuilder(s.Length + 16);
            foreach (char c in s)
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4")); else sb.Append(c);
                        break;
                }
            return sb.ToString();
        }
    }
}
