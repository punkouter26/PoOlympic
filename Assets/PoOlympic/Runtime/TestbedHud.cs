using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Minimal testbed HUD (Phase B6): FPS, sim time, control tick, pelvis height, lane state, and native-MuJoCo
    /// interaction buttons (shove, cube drop, reset). The broadcast HUD proper is Phase D.
    /// </summary>
    public class TestbedHud : MonoBehaviour
    {
        public PolicyRunner runner;
        public MjCubePool cubes;
        public string title = "PoOlympics — Testbed";
        public string version = "v0 · zero-brain";
        float _fps;

        void Update() => _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(1e-4f, Time.unscaledDeltaTime), 0.05f);

        unsafe void OnGUI()
        {
            var cam = Camera.main;
            var r = cam != null ? cam.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
            var area = new Rect(r.x, Screen.height - r.yMax, r.width, r.height); // GUI space is y-down
            GUI.Label(new Rect(area.x + 10, area.y + 8, 300, 24), title);
            string state = "—", z = "—";
            if (runner != null && runner.Initialized && MjScene.InstanceExists && MjScene.Instance.Data != null)
            {
                double h = runner.PelvisHeight(MjScene.Instance.Data);
                z = h.ToString("F3") + " m";
                state = h < 0.55 ? "FALLEN" : "STANDING";
            }
            GUI.Label(new Rect(area.x + area.width * 0.5f - 110, area.y + 8, 220, 60),
                $"FPS {_fps:F0}  |  t {Time.fixedTime:F2} s\ntick {(runner ? runner.ControlTick : 0)}  |  pelvis {z}\nlane 0: {state}");
            if (runner == null) return;
            if (GUI.Button(new Rect(area.x + 10, area.yMax - 44, 90, 34), "Reset")) runner.RequestReset();
            if (GUI.Button(new Rect(area.x + 106, area.yMax - 44, 90, 34), "Shove")) cubes.Shove(Random.insideUnitCircle.normalized * 0.5f);
            if (GUI.Button(new Rect(area.x + 202, area.yMax - 44, 90, 34), "Drop cube")) cubes.DropOnAthlete();
            GUI.Label(new Rect(area.xMax - 150, area.yMax - 30, 140, 24), version);
        }
    }
}
