using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Measures how hard the main camera moves in Play mode (calm camera, 2026-09-30): samples Camera.main every editor
    /// frame; Stop() reports the camera's angular / linear speed (p95, max) and the cuts (one-frame jumps &gt; 20° or
    /// &gt; 1 m). Driven from the Unity CLI: CameraMotionProbe.Start(), play for a while, CameraMotionProbe.Stop().
    /// </summary>
    public static class CameraMotionProbe
    {
        static readonly List<(float t, Vector3 p, Quaternion q)> Samples = new();
        static int _frame = -1;

        public static string Start()
        {
            Samples.Clear();
            EditorApplication.update -= Sample;
            EditorApplication.update += Sample;
            return "sampling";
        }

        static void Sample()
        {
            if (!EditorApplication.isPlaying || EditorApplication.isPaused) return;
            var cam = Camera.main;
            if (cam == null) return;
            if (Time.frameCount == _frame) return;                 // one sample per rendered game frame
            _frame = Time.frameCount;
            Samples.Add((Time.unscaledTime, cam.transform.position, cam.transform.rotation));
        }

        /// <summary>Per-frame samples as CSV (t, dt, moved m, turned deg, x, y, z) for a closer look.</summary>
        public static string Dump(string projectRelativePath)
        {
            var sb = new System.Text.StringBuilder("t,dt,dp,da,x,y,z\n");
            for (int i = 1; i < Samples.Count; i++)
            {
                var (t0, p0, q0) = Samples[i - 1];
                var (t1, p1, q1) = Samples[i];
                sb.Append($"{t1:F4},{t1 - t0:F4},{Vector3.Distance(p0, p1):F4},{Quaternion.Angle(q0, q1):F3},{p1.x:F3},{p1.y:F3},{p1.z:F3}\n");
            }
            var path = System.IO.Path.Combine(System.IO.Directory.GetParent(Application.dataPath)!.FullName, projectRelativePath);
            System.IO.File.WriteAllText(path, sb.ToString());
            return path;
        }

        public static string Stop()
        {
            EditorApplication.update -= Sample;
            var ang = new List<float>();
            var lin = new List<float>();
            int cuts = 0, fast = 0;
            for (int i = 1; i < Samples.Count; i++)
            {
                var (t0, p0, q0) = Samples[i - 1];
                var (t1, p1, q1) = Samples[i];
                float dt = t1 - t0, da = Quaternion.Angle(q0, q1), dp = Vector3.Distance(p0, p1);
                if (dt <= 0f) continue;
                if (da > 20f || dp > 5f) { cuts++; continue; }         // one-frame jump = a cut (no motion to follow)
                ang.Add(da / dt);
                lin.Add(dp / dt);
                if (da / dt > 60f || dp / dt > 12f) fast++;            // whip pan / fly-through: the eye-straining kind
            }
            static float P(List<float> v, float q) { if (v.Count == 0) return 0f; var s = v.OrderBy(x => x).ToList(); return s[Mathf.Clamp((int)(q * (s.Count - 1)), 0, s.Count - 1)]; }
            float secs = Samples.Count > 1 ? Samples[^1].t - Samples[0].t : 0f;
            return $"{secs:F0} s, {Samples.Count} frames · turn deg/s p50 {P(ang, .5f):F1} p95 {P(ang, .95f):F1} max {P(ang, 1f):F1}" +
                   $" · move m/s p50 {P(lin, .5f):F2} p95 {P(lin, .95f):F2} max {P(lin, 1f):F2} · cuts {cuts}" +
                   $" · fast frames {fast} ({100f * fast / Mathf.Max(1, ang.Count):F1} %: > 60 deg/s or > 12 m/s)";
        }
    }
}
