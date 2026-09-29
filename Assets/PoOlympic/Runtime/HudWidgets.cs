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
}
