using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Event 1 — Iron Pedestal (DESIGN.md §1, Rung 0): survive 20 s on a 1 m × 1 m block under random-direction gusts
    /// (0.5 m/s pelvis Δv) and 2 kg cube drops, each every 3–5 s. Everything is native MuJoCo: disturbances go through
    /// PolicyRunner.Request, elimination is read from mjData. The schedule runs on control ticks with a seeded RNG, so an
    /// attempt is reproducible (same seed ⇒ same gusts ⇒ same outcome).
    ///   Ready (countdown) → Live → Result → (auto) reset → Ready
    /// Scene contract (training/assets/scene_pedestal.xml): pedestal top at z = 0, ground at z = −0.5.
    /// </summary>
    public class IronPedestalEvent : MonoBehaviour
    {
        public enum Phase { Ready, Live, Result }

        public PolicyRunner runner;
        public MjCubePool cubes;
        [Header("Rules")]
        public float countdownSeconds = 3f;
        public float durationSeconds = 20f;
        public float gustDv = 0.5f;
        public Vector2 gustIntervalSeconds = new(3f, 5f);
        public Vector2 cubeIntervalSeconds = new(3f, 5f);
        public float cubeDropHeight = 1.5f;
        public float resultHoldSeconds = 4f;
        public bool autoRestart = true;
        [Tooltip("Attempt k uses seed + k.")]
        public int seed = 1;

        [Header("Elimination (DESIGN §1 fall rule)")]
        public double fallPelvisZ = 0.55;
        public double fallTiltDeg = 60;
        [Tooltip("A foot/toe geom centre this far below the pedestal top = stepped off.")]
        public double steppedOffZ = -0.06;

        public Phase Current { get; private set; } = Phase.Ready;
        public int Attempt { get; private set; }
        public float PhaseTime { get; private set; }
        public float LiveTime { get; private set; }
        public int Gusts { get; private set; }
        public int Cubes { get; private set; }
        public string Outcome { get; private set; } = "";
        public float BestTime { get; private set; }
        public int Wins { get; private set; }
        public event Action<IronPedestalEvent> OnResult;

        System.Random _rng;
        int _liveStartTick, _nextGustTick, _nextCubeTick;
        AthleteJudge _judge;

        float TickSeconds => (float)(runner.Contract.timestep * runner.Contract.decimation);
        int Ticks(float s) => Mathf.RoundToInt(s / TickSeconds);

        void Start() => BeginAttempt(first: true);

        public void Restart() => BeginAttempt(first: false);

        void BeginAttempt(bool first)
        {
            if (!first) Attempt++;
            _rng = new System.Random(seed + Attempt);
            if (!first) runner.RequestReset();
            Current = Phase.Ready;
            PhaseTime = LiveTime = 0;
            Gusts = Cubes = 0;
            Outcome = "";
        }

        double Uniform(Vector2 range) => range.x + _rng.NextDouble() * (range.y - range.x);

        unsafe void Update()
        {
            if (runner == null || !runner.Initialized || !MjScene.InstanceExists || MjScene.Instance.Data == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            _judge ??= new AthleteJudge(m, runner, "ground", "pedestal") { fallPelvisZ = fallPelvisZ, fallTiltDeg = fallTiltDeg, steppedOffZ = steppedOffZ };
            PhaseTime += Time.deltaTime;
            switch (Current)
            {
                case Phase.Ready:
                    if (PhaseTime >= countdownSeconds)
                    {
                        Current = Phase.Live;
                        PhaseTime = 0;
                        _liveStartTick = runner.ControlTick;
                        _nextGustTick = _liveStartTick + Ticks((float)Uniform(gustIntervalSeconds));
                        _nextCubeTick = _liveStartTick + Ticks((float)Uniform(cubeIntervalSeconds));
                    }
                    break;
                case Phase.Live:
                    LiveTime = (runner.ControlTick - _liveStartTick) * TickSeconds;
                    if (runner.ControlTick >= _nextGustTick)
                    {
                        double a = _rng.NextDouble() * 2 * Math.PI;
                        cubes.Shove(runner, new Vector2((float)Math.Cos(a), (float)Math.Sin(a)) * gustDv);
                        Gusts++;
                        _nextGustTick += Ticks((float)Uniform(gustIntervalSeconds));
                    }
                    if (runner.ControlTick >= _nextCubeTick)
                    {
                        cubes.DropOnAthlete(runner, cubeDropHeight);
                        Cubes++;
                        _nextCubeTick += Ticks((float)Uniform(cubeIntervalSeconds));
                    }
                    var why = _judge.Eliminated(m, d);
                    if (why != null) Finish(why, won: false);
                    else if (LiveTime >= durationSeconds) Finish("SURVIVED", won: true);
                    break;
                case Phase.Result:
                    if (autoRestart && PhaseTime >= resultHoldSeconds) Restart();
                    break;
            }
        }

        void Finish(string outcome, bool won)
        {
            Outcome = outcome;
            Current = Phase.Result;
            PhaseTime = 0;
            if (won) Wins++;
            BestTime = Mathf.Max(BestTime, LiveTime);
            OnResult?.Invoke(this);
            Debug.Log($"[IronPedestal] attempt {Attempt} seed {seed + Attempt}: {outcome} at {LiveTime:F2} s ({Gusts} gusts, {Cubes} cubes)");
        }
    }
}
