using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoOlympic
{
    /// <summary>
    /// The meet chosen in the main menu: the athlete in each of the 8 lanes and the event. Survives scene loads (static).
    /// Event scenes are built for a fixed lineup (Event 1: MATT + zombie alternating, the others 8 MATTs); the menu shows
    /// each event's lineup.
    /// </summary>
    public static class MeetLineup
    {
        public const string MenuScene = "MainMenu";
        public static readonly string[] Roster = { "MATT", "ZOMBIE" };
        /// <summary>One-line card stats per roster athlete (body + brains).</summary>
        public static string Stats(string athlete) => athlete switch
        {
            "MATT" => "1.84 m · 80 kg\nRung 0 + Rung 2",
            "ZOMBIE" => "1.10 m · 22 kg\nRung 0 (Event 1)",
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
}
