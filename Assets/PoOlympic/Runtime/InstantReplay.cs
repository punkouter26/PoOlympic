using System;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Instant replay of the end of a heat (owned by BroadcastHud). While the heat is live, every control tick's qpos of
    /// the whole scene goes into a ring (<see cref="Seconds"/> s at 50 Hz ≈ 0.6 MB for 8 athletes). Playback freezes the
    /// simulation (Time.timeScale 0: no mj_step, so nothing the brains or the judges see changes), writes the recorded
    /// poses back with mj_kinematics at <see cref="Speed"/> and lets the plug-in move the transforms; at the end the
    /// saved state is restored bit for bit and time runs again. Render only: no ctrl, no qvel, no contact is touched.
    /// </summary>
    public sealed unsafe class InstantReplay
    {
        public const float Seconds = 5f, Speed = 0.5f;

        float[] _ring;
        double[] _saved;
        int _nq, _frames, _head, _count, _substep, _decimation = 4;
        float _tickS = 0.02f, _t;
        bool _hooked;

        public bool Playing { get; private set; }
        public bool HasFootage => _count > 25 && !Playing;
        /// <summary>0..1 through the replay.</summary>
        public float Progress => _count > 1 ? Mathf.Clamp01(_t / ((_count - 1) * _tickS)) : 1f;

        /// <summary>Record while <paramref name="live"/> (BroadcastHud: the live phase and the first moments of the result).</summary>
        public bool Recording { get; set; }

        public void Hook(int decimation, float tickSeconds)
        {
            if (_hooked || !MjScene.InstanceExists) return;
            _decimation = Mathf.Max(1, decimation);
            _tickS = tickSeconds;
            MjScene.Instance.postUpdateEvent += OnStep;
            _hooked = true;
        }

        public void Unhook()
        {
            if (Playing) Stop();
            if (_hooked && MjScene.InstanceExists) MjScene.Instance.postUpdateEvent -= OnStep;
            _hooked = false;
        }

        public void Clear() { _head = _count = 0; }

        void OnStep(object sender, MjStepArgs e)
        {
            if (!Recording || Playing || ++_substep % _decimation != 0) return;
            int nq = (int)e.model->nq;
            if (_ring == null || nq != _nq)
            {
                _nq = nq;
                _frames = Mathf.CeilToInt(Seconds / _tickS);
                _ring = new float[_frames * nq];
                _saved = new double[nq];
                Clear();
            }
            int at = _head * nq;
            for (int i = 0; i < nq; i++) _ring[at + i] = (float)e.data->qpos[i];
            _head = (_head + 1) % _frames;
            _count = Math.Min(_count + 1, _frames);
        }

        public void Play()
        {
            if (!HasFootage || !MjScene.InstanceExists) return;
            var d = MjScene.Instance.Data;
            for (int i = 0; i < _nq; i++) _saved[i] = d->qpos[i];
            _t = 0f;
            Playing = true;
            Time.timeScale = 0f;
        }

        /// <summary>Per rendered frame while playing: the pose at the replay clock. Returns false when it has ended.</summary>
        public bool Step()
        {
            if (!Playing) return false;
            if (!MjScene.InstanceExists) { Playing = false; return false; }
            Time.timeScale = 0f;        // stay frozen: a hit-stop that began before the replay ends by restoring the speed (ImpactFx)
            _t +=Time.unscaledDeltaTime * Speed;
            float f = _t / _tickS;
            if (f >= _count - 1) { Stop(); return false; }
            int k = (int)f;
            float u = f - k;
            int first = (_head - _count + _frames) % _frames;
            int a = (first + k) % _frames * _nq, b = (first + k + 1) % _frames * _nq;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            // linear blend of neighbouring ticks (20 ms apart: quaternions barely move), then renormalise them
            for (int i = 0; i < _nq; i++) d->qpos[i] = _ring[a + i] + (_ring[b + i] - _ring[a + i]) * u;
            MujocoLib.mj_normalizeQuat(m, d->qpos);
            MujocoLib.mj_kinematics(m, d);
            MjScene.Instance.SyncUnityToMjState();
            return true;
        }

        public void Stop()
        {
            if (!Playing) return;
            Playing = false;
            if (MjScene.InstanceExists)
            {
                var m = MjScene.Instance.Model;
                var d = MjScene.Instance.Data;
                for (int i = 0; i < _nq; i++) d->qpos[i] = _saved[i];
                MujocoLib.mj_forward(m, d);
                MjScene.Instance.SyncUnityToMjState();
            }
            Time.timeScale = GameSettings.SimSpeed;
        }
    }
}
