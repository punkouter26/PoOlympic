using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>C# lane-keeping steering == training/poolympic/contract.py::steer_yaw_rate (vectors computed in Python).</summary>
    public class SteeringTests
    {
        static readonly double[][] Cases =
        {
            // w, x, y, z, lane offset y, vx command, expected wz
            new[] { 1, 0, 0, 0, 0.0, 1.0, 0.0 },
            new[] { 0.9987502603949663, 0, 0, 0.04997916927067833, 0.0, 1.0, -0.20000000000000018 },
            new[] { 0.9987502603949663, 0, 0, -0.04997916927067833, 0.5, 1.0, -0.0977798952189941 },
            new[] { 0.9950041652780258, 0.09983341664682815, 0, 0, -1.2, 2.0, 0.5 },
            new[] { 0.0707372016677029, 0, 0, 0.9974949866040544, 0.3, 0.0, -0.5 },
            new[] { 0.12050057851305121, 0.006030054809033824, 0, -0.9926949425765299, -0.4, 1.0, 0.5 },
            new[] { 0.9999500004166653, 0, 0, 0.009999833334166664, 4.0, 1.0, -0.5 },
            new[] { 0.9987502603949663, 0, 0, 0.04997916927067833, 0.5, -1.5, 0.0977798952189941 },
            new[] { 1, 0, 0, 0, -0.8, -0.5, -0.4710899614417263 },
        };

        [Test]
        public void SteerYawRateMatchesPython()
        {
            var c = Contract.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "..", "parity", "contract.json")));
            Assert.That(c.steering, Is.Not.Null, "contract.json has no steering block — re-export the contract");
            foreach (var k in Cases)
                Assert.That(c.SteerYawRate(k[0], k[1], k[2], k[3], k[4], k[5]), Is.EqualTo(k[6]).Within(1e-12), $"case {string.Join(", ", k)}");
        }
    }
}
