using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Screenshot helpers for the 9:16 broadcast frame: pins the Game view to a fixed portrait resolution (internal
    /// GameViewSizes API via reflection — there is no public one) and writes Play-mode screenshots outside Assets/.
    /// </summary>
    public static class CaptureTools
    {
        const string PortraitLabel = "PoOlympic Portrait";

        [MenuItem("PoOlympic/Tools/Game View 1080x1920 (portrait)")]
        public static string SetPortrait() => SetGameViewSize(1080, 1920);

        /// <summary>Select (adding if needed) a fixed-resolution size in the Game view's current platform group.</summary>
        public static string SetGameViewSize(int width, int height)
        {
            var asm = typeof(EditorWindow).Assembly;
            var sizesType = asm.GetType("UnityEditor.GameViewSizes");
            var singleton = typeof(ScriptableSingleton<>).MakeGenericType(sizesType);
            var sizes = singleton.GetProperty("instance").GetValue(null);
            var group = sizesType.GetProperty("currentGroup").GetValue(sizes);
            var groupType = group.GetType();
            int count = (int)groupType.GetMethod("GetTotalCount").Invoke(group, null);
            int index = -1;
            for (int i = 0; i < count; i++)
            {
                var size = groupType.GetMethod("GetGameViewSize").Invoke(group, new object[] { i });
                var st = size.GetType();
                if ((int)st.GetProperty("width").GetValue(size) == width && (int)st.GetProperty("height").GetValue(size) == height
                    && st.GetProperty("sizeType").GetValue(size).ToString() == "FixedResolution")
                {
                    index = i;
                    break;
                }
            }
            if (index < 0)
            {
                var sizeType = asm.GetType("UnityEditor.GameViewSize");
                var kind = asm.GetType("UnityEditor.GameViewSizeType");
                var ctor = sizeType.GetConstructor(new[] { kind, typeof(int), typeof(int), typeof(string) });
                var added = ctor.Invoke(new[] { Enum.Parse(kind, "FixedResolution"), width, height, PortraitLabel });
                groupType.GetMethod("AddCustomSize").Invoke(group, new[] { added });
                index = count;
            }
            var gv = EditorWindow.GetWindow(asm.GetType("UnityEditor.GameView"), false, "Game", false);
            var sel = gv.GetType().GetMethod("SizeSelectionCallback", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            sel.Invoke(gv, new object[] { index, null });
            gv.Repaint();
            return $"Game view size #{index}: {width}x{height}";
        }

        /// <summary>Play mode: Game view screenshot (incl. overlay UI) to a path relative to the project root, e.g.
        /// parity/d3/foo.png. Written at the end of the frame by Unity.</summary>
        public static string Screenshot(string projectRelativePath)
        {
            if (!EditorApplication.isPlaying) throw new InvalidOperationException("enter Play mode first");
            var full = Path.Combine(Path.GetDirectoryName(Application.dataPath), projectRelativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(full));
            ScreenCapture.CaptureScreenshot(full);
            return full;
        }
    }
}
