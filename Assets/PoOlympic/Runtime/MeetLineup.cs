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
}
