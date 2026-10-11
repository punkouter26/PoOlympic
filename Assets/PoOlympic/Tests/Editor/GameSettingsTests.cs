using System.Linq;
using NUnit.Framework;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>Player options and the spectator controls that do not need a running scene.</summary>
    public class GameSettingsTests
    {
        [TearDown] public void TearDown() { GameSettings.SimSpeed = 1f; Time.timeScale = 1f; }

        [Test]
        public void SpeedCyclesThroughEveryStepAndBackToRealTime()
        {
            GameSettings.SimSpeed = 1f;
            var seen = Enumerable.Range(0, GameSettings.Speeds.Length).Select(_ => { GameSettings.NextSpeed(); return GameSettings.SimSpeed; }).ToArray();
            CollectionAssert.AreEquivalent(GameSettings.Speeds, seen);
            Assert.AreEqual(1f, GameSettings.SimSpeed, "a full cycle ends at real time");
            Assert.AreEqual(1f, Time.timeScale);
        }

        [Test]
        public void SpeedIsNotAppliedWhileTimeIsFrozen()
        {
            Time.timeScale = 0f;                                    // a hit-stop or a replay owns the clock
            GameSettings.SimSpeed = 0.5f;
            Assert.AreEqual(0f, Time.timeScale, "the freeze must survive a speed change");
            Assert.AreEqual(0.5f, GameSettings.SimSpeed, "…and release to the new speed");
        }
    }
}
