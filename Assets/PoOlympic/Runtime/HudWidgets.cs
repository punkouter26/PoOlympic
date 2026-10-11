using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoOlympic
{
    /// <summary>UI Toolkit sparkline (Painter2D): a 0..1 series left to right, a dashed 50 % guide, colour by the last value
    /// (green high, amber middle, red low). Used for the brain-confidence column and the stats card.</summary>
    public class Sparkline : VisualElement
    {
        readonly List<float> _values = new();
        public float lineWidth = 2.5f;
        public int capacity = 60;

        static readonly Color High = new(0.35f, 0.9f, 0.55f), Mid = new(1f, 0.8f, 0.25f), Low = new(1f, 0.35f, 0.35f);
        /// <summary>"No value" text colour (= --po-muted).</summary>
        public static readonly Color Muted = new(0.55f, 0.6f, 0.69f);

        public Sparkline()
        {
            AddToClassList("bh-spark");
            generateVisualContent += Draw;
            pickingMode = PickingMode.Ignore;
        }

        public static Color ColorOf(float v) => v >= 0.7f ? High : v >= 0.4f ? Color.Lerp(Mid, High, (v - 0.4f) / 0.3f) : Color.Lerp(Low, Mid, v / 0.4f);

        public void SetValues(IReadOnlyList<float> values)
        {
            _values.Clear();
            int start = Mathf.Max(0, values.Count - capacity);
            for (int i = start; i < values.Count; i++) _values.Add(values[i]);
            MarkDirtyRepaint();
        }

        void Draw(MeshGenerationContext ctx)
        {
            var r = contentRect;
            if (r.width < 4 || r.height < 4) return;
            var p = ctx.painter2D;
            p.lineWidth = 1f;
            p.strokeColor = new Color(1f, 1f, 1f, 0.18f);
            p.BeginPath();
            float mid = r.yMin + r.height * 0.5f;
            for (float x = r.xMin; x < r.xMax; x += 8f) { p.MoveTo(new Vector2(x, mid)); p.LineTo(new Vector2(Mathf.Min(x + 4f, r.xMax), mid)); }
            p.Stroke();
            if (_values.Count < 2) return;
            p.lineWidth = lineWidth;
            p.lineJoin = LineJoin.Round;
            p.strokeColor = ColorOf(_values[_values.Count - 1]);
            p.BeginPath();
            for (int i = 0; i < _values.Count; i++)
            {
                float x = r.xMin + r.width * i / (capacity - 1);
                float y = r.yMax - Mathf.Clamp01(_values[i]) * r.height;
                if (i == 0) p.MoveTo(new Vector2(x, y)); else p.LineTo(new Vector2(x, y));
            }
            p.Stroke();
        }
    }

    /// <summary>Athlete status at a glance, two bars and no text (standings rows): top = balance (full and green = the
    /// centre of mass well inside the feet, short and red = going over), bottom = effort (how hard the joints work),
    /// shading from cyan to red as the athlete tires.</summary>
    public class StatusBars : VisualElement
    {
        float _balance = 1f, _effort, _fatigue;
        static readonly Color Track = new(1f, 1f, 1f, 0.14f), Fresh = new(0.47f, 0.86f, 1f), Spent = new(1f, 0.35f, 0.35f);

        public StatusBars()
        {
            AddToClassList("bh-status");
            generateVisualContent += Draw;
        }

        public void Set(float balance, float effort, float fatigue)
        {
            balance = Mathf.Clamp01(balance); effort = Mathf.Clamp01(effort); fatigue = Mathf.Clamp01(fatigue);
            if (Mathf.Abs(balance - _balance) < 0.02f && Mathf.Abs(effort - _effort) < 0.02f && Mathf.Abs(fatigue - _fatigue) < 0.02f) return;
            _balance = balance; _effort = effort; _fatigue = fatigue;
            MarkDirtyRepaint();
        }

        void Draw(MeshGenerationContext ctx)
        {
            var r = contentRect;
            if (r.width < 8 || r.height < 8) return;
            float h = r.height * 0.3f;
            Bar(ctx.painter2D, r.xMin, r.yMin + r.height * 0.12f, r.width, h, _balance, Sparkline.ColorOf(_balance));
            Bar(ctx.painter2D, r.xMin, r.yMax - r.height * 0.12f - h, r.width, h, _effort, Color.Lerp(Fresh, Spent, _fatigue));
        }

        static void Bar(Painter2D p, float x, float y, float w, float h, float value, Color fill)
        {
            p.fillColor = Track;
            p.BeginPath(); p.MoveTo(new Vector2(x, y)); p.LineTo(new Vector2(x + w, y)); p.LineTo(new Vector2(x + w, y + h)); p.LineTo(new Vector2(x, y + h)); p.ClosePath(); p.Fill();
            if (value <= 0.01f) return;
            p.fillColor = fill;
            float v = w * value;
            p.BeginPath(); p.MoveTo(new Vector2(x, y)); p.LineTo(new Vector2(x + v, y)); p.LineTo(new Vector2(x + v, y + h)); p.LineTo(new Vector2(x, y + h)); p.ClosePath(); p.Fill();
        }
    }
}
