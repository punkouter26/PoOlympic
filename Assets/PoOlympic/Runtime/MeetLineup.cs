using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic
{
    /// <summary>
    /// The meet chosen in the main menu: the athlete in each of the 8 lanes and the event. Survives scene loads (static).
    /// Event scenes are roster scenes (every roster body in every lane); LaneLineup keeps the picked body per lane.
    /// </summary>
    public static class MeetLineup
    {
        public const string MenuScene = "MainMenu";
        public static readonly string[] Roster = { "MATT", "ZOMBIE" };
        /// <summary>One-line card stats per roster athlete (body + brains).</summary>
        public static string Stats(string athlete) => athlete switch
        {
            "MATT" => "1.84 m · 80 kg\nRung 0/2 · Crawl",
            "ZOMBIE" => "1.10 m · 22 kg\nRung 0/2 · Crawl",
            _ => "",
        };
        public static string[] Athletes { get; private set; } = Enumerable.Repeat(Roster[0], 8).ToArray();
        public static int EventNumber { get; private set; } = 1;

        public static void Set(string[] athletes, int eventNumber)
        {
            Athletes = athletes.ToArray();
            EventNumber = eventNumber;
        }

        public static bool MenuAvailable => Application.CanStreamedLevelBeLoaded(MenuScene);

        public static void ReturnToMenu()
        {
            if (MenuAvailable) SceneManager.LoadScene(MenuScene);
        }
    }

    /// <summary>
    /// Gauntlet (game layer): a series of events played back to back with the menu's lineup; one heat per event,
    /// points per place (10-8-6-5-4-3-2-1) summed per lane. Survives scene loads (static); BroadcastHud awards the
    /// points on each result card and moves on with Next().
    /// </summary>
    public static class Gauntlet
    {
        public static readonly int[] PointsTable = { 10, 8, 6, 5, 4, 3, 2, 1 };
        public static int[] Events { get; private set; } = Array.Empty<int>();
        static string[] _scenes = Array.Empty<string>();
        public static int Index { get; private set; }
        public static bool Active => Index < Events.Length;
        public static bool IsLast => Index == Events.Length - 1;
        /// <summary>The current stage's heat has been scored (further heats of this scene wait for Next()).</summary>
        public static bool CurrentHeatPlayed { get; private set; }
        static readonly int[] _points = new int[8];
        /// <summary>Final table of the last completed gauntlet (menu), empty if none.</summary>
        public static string LastResult { get; private set; } = "";

        public static void Begin(int[] events, string[] scenes)
        {
            Events = events.ToArray();
            _scenes = scenes.ToArray();
            Index = 0;
            CurrentHeatPlayed = false;
            Array.Clear(_points, 0, _points.Length);
            Load();
        }

        static void Load()
        {
            MeetLineup.Set(MeetLineup.Athletes, Events[Index]);
            SceneManager.LoadScene(_scenes[Index]);
        }

        /// <summary>Lane number (1-8) from a HUD name such as "L3", "Z3", "S8", "M1".</summary>
        public static int LaneOf(string name)
        {
            var digits = new string(name.Where(char.IsDigit).ToArray());
            return int.TryParse(digits, out var k) ? k : 0;
        }

        public static void Award(IEnumerable<(string name, int place)> results)
        {
            foreach (var (name, place) in results)
            {
                int lane = LaneOf(name);
                if (lane >= 1 && lane <= 8 && place >= 1 && place <= PointsTable.Length) _points[lane - 1] += PointsTable[place - 1];
            }
            CurrentHeatPlayed = true;
        }

        public static IEnumerable<(string lane, int points)> Table() =>
            Enumerable.Range(0, 8).OrderByDescending(k => _points[k])
                .Select(k => ($"Lane {k + 1} {MeetLineup.Athletes[k]}", _points[k]));

        public static void Next()
        {
            if (IsLast)
            {
                LastResult = string.Join("\n", Table().Take(3).Select((x, i) => $"{i + 1}. {x.lane} — {x.points} pts"));
                Abandon();
                MeetLineup.ReturnToMenu();
                return;
            }
            Index++;
            CurrentHeatPlayed = false;
            Load();
        }

        public static void Abandon()
        {
            Events = Array.Empty<int>();
            Index = 0;
            CurrentHeatPlayed = false;
        }
    }
}
