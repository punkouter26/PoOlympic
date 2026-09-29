using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Feature 6 — "Hawk-Eye" balance overlay for one athlete, drawn on the ground from AthleteTelemetry (MuJoCo state):
    ///   support polygon  outline of the convex hull of everything touching a surface (feet; hands and knees on all fours)
    ///   CoM ring         the centre of mass projected onto the support height
    ///   plumb line       from the centre of mass straight down to the ring
    /// Colour = balance margin: green ≥ safeMargin inside, amber near the edge, red outside the polygon (off balance).
    /// Visible always in stationary events (showAlways), otherwise only while the athlete is the TensionMeter's hot
    /// athlete in real danger. All three are LineRenderers under the athlete (built in the scene by the FX upgrade).
    /// </summary>
    [DefaultExecutionOrder(160)]
    public class BalanceOverlay : MonoBehaviour
    {
        public AthleteTelemetry telemetry;
        public LineRenderer hull, plumb, ring;
        public bool showAlways;
        public float safeMargin = 0.06f;
        public float ringRadius = 0.1f;
        public float lift = 0.012f;

        static readonly Color Safe = new(0.35f, 0.95f, 0.55f), Edge = new(1f, 0.78f, 0.2f), Out = new(1f, 0.3f, 0.3f);
        const int RingPoints = 20;
        float _alpha;

        void LateUpdate()
        {
            var t = telemetry;
            bool want = t != null && t.Ready && !float.IsNaN(t.BalanceMargin) && t.SupportHull.Count > 0;
            if (want && !showAlways)
            {
                var tm = TensionMeter.Instance;
                want = tm != null && tm.Hot == t.Runner && tm.HotDanger >= 0.45f;
            }
            _alpha = Mathf.MoveTowards(_alpha, want ? 1f : 0f, Time.unscaledDeltaTime * 4f);
            bool on = _alpha > 0.01f && t != null && t.SupportHull.Count > 0;
            hull.enabled = plumb.enabled = ring.enabled = on;
            if (!on) return;

            float m = t.BalanceMargin;
            var c = float.IsNaN(m) ? Edge : m >= safeMargin ? Safe : m >= 0f ? Color.Lerp(Edge, Safe, m / safeMargin) : Color.Lerp(Edge, Out, Mathf.Clamp01(-m / 0.04f));
            c.a = _alpha;
            float y = t.SupportHeight + lift;

            int n = t.SupportHull.Count;
            if (n == 1)
            {
                hull.loop = true;
                hull.positionCount = RingPoints;
                var p = t.SupportHull[0];
                for (int i = 0; i < RingPoints; i++)
                {
                    float a = i * Mathf.PI * 2f / RingPoints;
                    hull.SetPosition(i, new Vector3(p.x + Mathf.Cos(a) * 0.03f, y, p.z + Mathf.Sin(a) * 0.03f));
                }
            }
            else
            {
                hull.loop = n > 2;
                hull.positionCount = n;
                for (int i = 0; i < n; i++) { var p = t.SupportHull[i]; hull.SetPosition(i, new Vector3(p.x, y, p.z)); }
            }
            var com = t.Com;
            var foot = new Vector3(com.x, y, com.z);
            plumb.positionCount = 2;
            plumb.SetPosition(0, com);
            plumb.SetPosition(1, foot);
            ring.loop = true;
            ring.positionCount = RingPoints;
            for (int i = 0; i < RingPoints; i++)
            {
                float a = i * Mathf.PI * 2f / RingPoints;
                ring.SetPosition(i, foot + new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a)) * ringRadius);
            }
            var faint = new Color(c.r, c.g, c.b, c.a * 0.45f);
            hull.startColor = hull.endColor = c;
            ring.startColor = ring.endColor = c;
            plumb.startColor = faint;
            plumb.endColor = c;
        }
    }
}
