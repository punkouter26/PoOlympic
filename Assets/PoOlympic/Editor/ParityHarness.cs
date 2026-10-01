using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Editor
{
    /// <summary>
    /// Parity gates G2–G4 (DESIGN.md §4) against a golden CPU-MuJoCo reference recorded by
    /// training/tools/record_reference.py. Runs in Edit mode on the Unity-compiled model + native library:
    ///   G2  C# ObservationBuilder vs recorded obs (state injected from the reference)   max |Δ| < 1e-5
    ///   G3  Inference Engine on recorded obs vs recorded action_raw / ctrl               max |Δ| < 1e-4
    ///   G4  recorded ctrl + disturbances replayed open-loop through mj_step             max |Δqpos| < 1e-3 over 1 s
    /// </summary>
    public static unsafe class ParityHarness
    {
        public const string TestbedScene = "Assets/PoOlympic/Scenes/Testbed_ZeroBrain.unity";
        /// <summary>Solo testbed per athlete body (Phase Z7): the scene whose compiled model the body's references
        /// were recorded on (training/assets/scene_&lt;body&gt;.xml).</summary>
        public static string TestbedSceneOf(string body) =>
            body == "matt" ? TestbedScene : body == "mattbio" ? StanceTestbed.ScenePath : SoloTestbed.ScenePathOf(body);

        /// <summary>Unity scene holding the reference's MJCF: the solo pedestal (Event 1 practice scene) or the body's testbed.</summary>
        public static string SceneFor(RefMeta meta, string body) =>
            meta.scene == "scene_pedestal.xml" ? EventScenes.IronPedestalScene : TestbedSceneOf(body);
        public const string ModelsFolder = "Assets/PoOlympic/Models";
        public const double G2Tol = 1e-5, G3Tol = 1e-4, G4Tol = 1e-3;
        public const int G4Ticks = 50;

        [Serializable] public class RefJoint { public string name; public int type; public int qposadr; public int dofadr; }
        [Serializable] public class RefMeta { public string body; public string onnx; public string fingerprint_sha256; public double[] command; public RefJoint[] joints; public int nq; public int nv;
                                              public int contract_version; public double[] skill;   // v4 references: stance-skill block
                                              public string scene; }                                // MJCF the reference was rolled out on
        [Serializable] public class RefFrame { public int tick; public double t; public double phase; public double[] qpos; public double[] qvel; public double[] obs; public double[] action_raw; public double[] ctrl; public double[] actuator_force; }
        [Serializable] public class Reference { public RefMeta meta; public Disturbance[] disturbances; public RefFrame[] frames; }

        public class Report
        {
            public string name;
            public double g2MaxAbs, g3MaxAbsAction, g3MaxAbsCtrl, g4MaxAbs1s, g4MaxAbs5s;
            public int g4FirstTickOver = -1;
            public bool G2 => g2MaxAbs < G2Tol;
            public bool G3 => g3MaxAbsAction < G3Tol && g3MaxAbsCtrl < G3Tol;
            public bool G4 => g4MaxAbs1s < G4Tol;
        }

        static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

        /// <summary>Copy parity/contract.json and parity/brains/* into Assets (verbatim).</summary>
        [MenuItem("PoOlympic/Parity/Sync Contract + Brains")]
        public static void SyncArtifacts()
        {
            var models = Path.Combine(ProjectRoot, ModelsFolder);
            Directory.CreateDirectory(Path.Combine(models, "Brains"));
            File.Copy(Path.Combine(ProjectRoot, "parity", "contract.json"), Path.Combine(models, "contract.json"), true);
            foreach (var f in Directory.GetFiles(Path.Combine(ProjectRoot, "parity"), "contract_*.json"))   // other bodies
                File.Copy(f, Path.Combine(models, Path.GetFileName(f)), true);
            foreach (var f in Directory.GetFiles(Path.Combine(ProjectRoot, "parity", "brains")))
                File.Copy(f, Path.Combine(models, "Brains", Path.GetFileName(f)), true);
            AssetDatabase.Refresh();
        }

        public static Contract LoadContract(string body = "matt") =>
            Contract.Parse(File.ReadAllText(Path.Combine(ProjectRoot, ModelsFolder, body == "matt" ? "contract.json" : $"contract_{body}.json")));

        public static Reference LoadReference(string name) =>
            JsonUtility.FromJson<Reference>(File.ReadAllText(Path.Combine(ProjectRoot, "parity", $"reference_trajectory_{name}.json")));

        public static Report Run(string name, bool openScene = true)
        {
            var reference = LoadReference(name);
            var body = string.IsNullOrEmpty(reference.meta.body) ? "matt" : reference.meta.body;
            if (openScene) EditorSceneManager.OpenScene(SceneFor(reference.meta, body), OpenSceneMode.Single);
            var contract = LoadContract(body);
            if (reference.meta.fingerprint_sha256 != contract.fingerprint_sha256)
                throw new InvalidOperationException("reference was recorded on a different model than the contract");

            foreach (var stale in UnityEngine.Object.FindObjectsByType<MjScene>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                UnityEngine.Object.DestroyImmediate(stale.gameObject);
            var scene = MjScene.Instance;
            scene.CreateScene();
            var report = new Report { name = name };
            try
            {
                var m = scene.Model;
                var d = scene.Data;
                if (m->opt.timestep != contract.timestep) throw new InvalidOperationException($"timestep {m->opt.timestep:R}");
                var bind = new AthleteBinding(m, contract);
                var joints = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_JOINT, (int)m->njnt);
                var (qmap, vmap) = StateMaps(m, reference.meta, joints);
                var cmd = new Vector3((float)reference.meta.command[0], (float)reference.meta.command[1], (float)reference.meta.command[2]);
                var frames = reference.frames;
                int n = contract.num_actions;

                // ---- G2: observation builder (v4 references: + the stance-skill block, march cadence drives the clock)
                bool v4 = reference.meta.contract_version >= 4;
                if (v4 && !contract.SupportsSkills) throw new InvalidOperationException("v4 reference but the contract has no skill block");
                float[] skill = v4 ? Array.ConvertAll(reference.meta.skill, x => (float)x) : null;
                var obs = new float[contract.ObsDimFor(v4 ? 4 : 3)];
                var last = new float[n];
                foreach (var f in frames)
                {
                    InjectState(d, f, qmap, vmap);
                    ObservationBuilder.Build(contract, bind, d->qpos, d->qvel, cmd, f.phase, last, obs, skill);
                    for (int i = 0; i < obs.Length; i++) report.g2MaxAbs = Math.Max(report.g2MaxAbs, Math.Abs(obs[i] - f.obs[i]));
                    for (int i = 0; i < n; i++) last[i] = (float)f.action_raw[i];
                }

                // ---- G3: policy replay
                var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>($"{ModelsFolder}/Brains/{reference.meta.onnx}");
                if (asset == null) throw new FileNotFoundException($"brain {reference.meta.onnx} not synced into {ModelsFolder}/Brains");
                using (var brain = new PolicyBrain(asset, obs.Length))
                {
                    var ctrl = new float[n];
                    var act = new float[n];
                    foreach (var f in frames)
                    {
                        for (int i = 0; i < obs.Length; i++) obs[i] = (float)f.obs[i];
                        brain.Run(obs, ctrl, act);
                        for (int i = 0; i < n; i++)
                        {
                            report.g3MaxAbsAction = Math.Max(report.g3MaxAbsAction, Math.Abs(act[i] - f.action_raw[i]));
                            report.g3MaxAbsCtrl = Math.Max(report.g3MaxAbsCtrl, Math.Abs(ctrl[i] - f.ctrl[i]));
                        }
                    }
                }

                // ---- G4: open-loop physics replay
                var byTick = new Dictionary<int, List<Disturbance>>();
                foreach (var dist in reference.disturbances)
                {
                    if (!byTick.TryGetValue(dist.tick, out var l)) byTick[dist.tick] = l = new List<Disturbance>();
                    l.Add(dist);
                }
                MujocoLib.mj_resetData(m, d);
                InjectState(d, frames[0], qmap, vmap);
                MujocoLib.mj_forward(m, d);
                for (int k = 0; k < frames.Length; k++)
                {
                    double drift = 0;
                    for (int i = 0; i < qmap.Length; i++) drift = Math.Max(drift, Math.Abs(d->qpos[qmap[i]] - frames[k].qpos[i]));
                    if (k <= G4Ticks) report.g4MaxAbs1s = Math.Max(report.g4MaxAbs1s, drift);
                    report.g4MaxAbs5s = Math.Max(report.g4MaxAbs5s, drift);
                    if (drift >= G4Tol && report.g4FirstTickOver < 0) report.g4FirstTickOver = k;
                    for (int i = 0; i < n; i++) d->ctrl[bind.ActuatorIds[i]] = frames[k].ctrl[i];
                    if (byTick.TryGetValue(k, out var list)) foreach (var dist in list) dist.Apply(m, d, joints);
                    for (int s = 0; s < contract.decimation; s++) MujocoLib.mj_step(m, d);
                }
            }
            finally
            {
                scene.DestroyScene();
                UnityEngine.Object.DestroyImmediate(scene.gameObject);
            }
            WriteReport(report);
            return report;
        }

        /// <summary>Python qpos/qvel index i → Unity address, by joint name.</summary>
        static (int[] q, int[] v) StateMaps(MujocoLib.mjModel_* m, RefMeta meta, Dictionary<string, int> joints)
        {
            var q = new int[meta.nq];
            var v = new int[meta.nv];
            foreach (var j in meta.joints)
            {
                if (!joints.TryGetValue(j.name, out var uj)) throw new KeyNotFoundException($"joint {j.name} missing in Unity model");
                int nq = j.type == 0 ? 7 : j.type == 1 ? 4 : 1;
                int nv = j.type == 0 ? 6 : j.type == 1 ? 3 : 1;
                if (m->jnt_type[uj] != j.type) throw new InvalidOperationException($"joint {j.name} type differs");
                for (int i = 0; i < nq; i++) q[j.qposadr + i] = m->jnt_qposadr[uj] + i;
                for (int i = 0; i < nv; i++) v[j.dofadr + i] = m->jnt_dofadr[uj] + i;
            }
            return (q, v);
        }

        static void InjectState(MujocoLib.mjData_* d, RefFrame f, int[] qmap, int[] vmap)
        {
            for (int i = 0; i < qmap.Length; i++) d->qpos[qmap[i]] = f.qpos[i];
            for (int i = 0; i < vmap.Length; i++) d->qvel[vmap[i]] = f.qvel[i];
        }

        static void WriteReport(Report r)
        {
            string F(double x) => x.ToString("E3", CultureInfo.InvariantCulture);
            var sb = new StringBuilder();
            sb.Append("{\"reference\":\"").Append(r.name).Append("\",")
              .Append($"\"G2\":{{\"max_abs\":{F(r.g2MaxAbs)},\"tol\":{G2Tol:E0},\"pass\":{(r.G2 ? "true" : "false")}}},")
              .Append($"\"G3\":{{\"max_abs_action_raw\":{F(r.g3MaxAbsAction)},\"max_abs_ctrl\":{F(r.g3MaxAbsCtrl)},\"tol\":{G3Tol:E0},\"pass\":{(r.G3 ? "true" : "false")}}},")
              .Append($"\"G4\":{{\"max_abs_qpos_1s\":{F(r.g4MaxAbs1s)},\"max_abs_qpos_5s\":{F(r.g4MaxAbs5s)},\"first_tick_over_tol\":{r.g4FirstTickOver},\"tol\":{G4Tol:E0},\"pass\":{(r.G4 ? "true" : "false")}}},")
              .Append($"\"generated_utc\":\"{DateTime.UtcNow:o}\"}}");
            File.WriteAllText(Path.Combine(ProjectRoot, "parity", $"gate_report_{r.name}.json"), sb.ToString());
        }
    }
}
