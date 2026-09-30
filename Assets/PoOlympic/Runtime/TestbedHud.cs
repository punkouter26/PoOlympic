using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Minimal testbed HUD (Phase B6 / C6): FPS, sim time, control tick, per-lane pelvis height / distance / state, and
    /// native-MuJoCo interaction buttons (shove, cube drop, reset) acting on the selected lane. The broadcast HUD
    /// proper is Phase D. Shared anchors (HudAnchors): TL title + sim time / lanes · TC FPS · TR menu · BL debug
    /// controls · BR version.
    /// </summary>
    public class TestbedHud : MonoBehaviour
    {
        public PolicyRunner runner;
        [Tooltip("All lanes of a meet scene (empty in the solo testbed).")]
        public PolicyRunner[] lanes = new PolicyRunner[0];
        public MjCubePool cubes;
        public string title = "PoOlympics — Testbed";
        public string version = "v0 · zero-brain";
        float _fps;

        void Update() => _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(1e-4f, Time.unscaledDeltaTime), 0.05f);

        static unsafe string LaneLine(PolicyRunner r)
        {
            if (r == null || !r.Initialized || !MjScene.InstanceExists || MjScene.Instance.Data == null) return "—";
            var d = MjScene.Instance.Data;
            double h = r.PelvisHeight(d);
            double x = d->qpos[r.Binding.RootQposAdr] - r.laneOriginX;
            return $"{r.command.x:F2} m/s  x {x,5:F1} m  z {h:F2}  {(h < 0.55 ? "FALLEN" : "UP")}";
        }

        // C8: behaviour switching from the HUD — command presets for every lane (MuJoCo frame: vx, vy, wz)
        static readonly (string label, Vector3 cmd, bool laneKeep)[] Presets =
        {
            ("Stop", Vector3.zero, false), ("Walk", new Vector3(1f, 0f, 0f), true), ("Sprint", new Vector3(3f, 0f, 0f), true),
            ("Back", new Vector3(-1.5f, 0f, 0f), true), ("Crab ◀", new Vector3(0f, 0.75f, 0f), false),
            ("Crab ▶", new Vector3(0f, -0.75f, 0f), false), ("Spin", new Vector3(0f, 0f, 2.2f), false),
        };

        void CommandPresets(Rect area)
        {
            float w = Mathf.Min(78f, (area.width - 20f) / Presets.Length - 4f);
            for (int i = 0; i < Presets.Length; i++)
            {
                var (label, cmd, keep) = Presets[i];
                if (GUI.Button(new Rect(area.x + 10 + i * (w + 4), area.yMax - 84, w, 32), label))
                    foreach (var r in lanes)
                    {
                        if (r == null) continue;
                        r.command = cmd;
                        r.laneKeeping = keep;
                    }
            }
        }

        void OnGUI()
        {
            var cam = Camera.main;
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var area = new Rect(r.x, Screen.height - r.yMax, r.width, r.height); // GUI space is y-down
            GUI.Label(new Rect(area.x + 10, area.y + 8, 300, 24), title);
            GUI.Label(new Rect(area.x + 10, area.y + 30, 300, 24), $"t {Time.fixedTime:F2} s · tick {(runner ? runner.ControlTick : 0)}");
            GUI.Label(new Rect(area.x + area.width * 0.5f - 40, area.y + 8, 80, 24), $"FPS {_fps:F0}");
            if (MeetLineup.MenuAvailable && GUI.Button(new Rect(area.xMax - 90, area.y + 8, 80, 34), "Menu")) MeetLineup.ReturnToMenu();
            if (lanes.Length == 0)
                GUI.Label(new Rect(area.x + 10, area.y + 56, 320, 24), $"lane 0: {LaneLine(runner)}");
            else
                for (int i = 0; i < lanes.Length; i++)
                {
                    var style = lanes[i] == runner ? GUI.skin.box : GUI.skin.label;
                    GUI.Label(new Rect(area.x + 10, area.y + 56 + 20 * i, 320, 20), $"L{i}  {LaneLine(lanes[i])}", style);
                }
            if (runner == null) return;
            if (GUI.Button(new Rect(area.x + 10, area.yMax - 44, 80, 34), "Reset")) runner.RequestReset();
            if (GUI.Button(new Rect(area.x + 96, area.yMax - 44, 80, 34), "Shove")) cubes.Shove(runner, Random.insideUnitCircle.normalized * 0.5f);
            if (GUI.Button(new Rect(area.x + 182, area.yMax - 44, 80, 34), "Drop cube")) cubes.DropOnAthlete(runner);
            if (lanes.Length > 0 && GUI.Button(new Rect(area.x + 268, area.yMax - 44, 80, 34), $"Lane {System.Array.IndexOf(lanes, runner)} ▸"))
                runner = lanes[(System.Array.IndexOf(lanes, runner) + 1) % lanes.Length];
            if (lanes.Length > 0) CommandPresets(area);
            GUI.Label(new Rect(area.xMax - 150, area.yMax - 30, 140, 24), version);
        }
    }
}
