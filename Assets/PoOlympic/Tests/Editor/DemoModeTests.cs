using System.Linq;
using NUnit.Framework;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Demo mode (2026-09-30): random lineups and the season table (the scene loop is checked in Play mode).</summary>
    public class DemoModeTests
    {
        const string SeasonKey = "PoOlympic.DemoSeason";
        string _saved;

        [SetUp] public void SaveSeason() => _saved = PlayerPrefs.GetString(SeasonKey, null);

        [TearDown]
        public void RestoreSeason()
        {
            if (_saved == null) PlayerPrefs.DeleteKey(SeasonKey);
            else PlayerPrefs.SetString(SeasonKey, _saved);
        }

        [Test]
        public void LineupsFillAllEightLanesFromTheRoster()
        {
            var themes = new System.Collections.Generic.HashSet<string>();
            for (int i = 0; i < 400; i++)
            {
                var lanes = DemoLineups.Random(MeetLineup.Roster, out var theme);
                Assert.AreEqual(8, lanes.Length);
                Assert.IsTrue(lanes.All(a => MeetLineup.Roster.Contains(a)), string.Join(",", lanes));
                if (theme == "Mixed lineup") Assert.GreaterOrEqual(lanes.Distinct().Count(), 2, "a mixed lineup mixes bodies");
                themes.Add(theme.Split(':')[0].Split(' ')[0]);
            }
            CollectionAssert.IsSupersetOf(themes, new[] { "Mixed", "All", "4", "One" }, "every lineup kind is drawn");
        }

        [Test]
        public void OneBodyRosterAlwaysFillsWithThatBody()
        {
            var lanes = DemoLineups.Random(new[] { "MATT" }, out var theme);
            Assert.IsTrue(lanes.All(a => a == "MATT"));
            Assert.AreEqual("All MATT", theme);
        }

        [Test]
        public void SeasonCountsSeriesWinsAndPointsPerEvent()
        {
            DemoSeason.Reset();
            Assert.AreEqual("", DemoSeason.Summary());
            var lineup = new[] { "MATT", "MATT", "MATT", "MATT", "ZOMBIE", "ZOMBIE", "ZOMBIE", "ZOMBIE" };
            DemoSeason.Record(lineup, new[] { 10, 8, 6, 5, 20, 3, 2, 2 }, 2);   // lane 5 (ZOMBIE) wins the series
            Assert.AreEqual(1, DemoSeason.SeriesPlayed);
            StringAssert.Contains("wins ZOMBIE 1 · MATT 0", DemoSeason.Summary());
            StringAssert.Contains("MATT 3.6", DemoSeason.Summary());      // 29 points / (4 lanes × 2 events) = 3.625
            StringAssert.Contains("ZOMBIE 3.4", DemoSeason.Summary());    // 27 / 8 = 3.375
        }
    }
}
