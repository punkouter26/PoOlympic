using System;
using System.Collections.Generic;
using System.Linq;
using Mujoco;
using NUnit.Framework;
using PoOlympic.Editor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoOlympic.Tests
{
    /// <summary>
    /// Crowd events (2026-09-30): in every built event scene, every body part of every athlete collides with every body part
    /// of the athletes of every other lane — in the MuJoCo model Unity compiles from the scene (= training/tools/
    /// check_crowd_contacts.py on the MJCF). Per ordered pair of athletes in different lanes and per pair of parts: the
    /// contype/conaffinity filter passes, no &lt;exclude&gt; hides it, and with athlete B moved so that part b overlaps part a,
    /// mj_kinematics + mj_collision report a contact between exactly those two geoms. Roster scenes hold MATT (L&lt;k&gt;_) and
    /// the zombie (Z&lt;k&gt;_) in every lane; only one of them is ever active (LaneLineup switches the other off before MuJoCo
    /// compiles), so every body is checked against both bodies of every other lane and same-lane pairs are skipped.
    /// </summary>
    public unsafe class CrowdContactTests
    {
        static IEnumerable<string> EventScenes() => PoOlympic.Editor.EventScenes.EventScenePaths.Values.Distinct().OrderBy(p => p);

        [TestCaseSource(nameof(EventScenes))]
        public void EveryPartCollidesWithEveryPartOfTheOtherLanes(string scenePath)
        {
            EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            var scene = MjScene.Instance;
            scene.CreateScene();
            try
            {
                var m = scene.Model;
                var d = scene.Data;
                var geomName = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom).ToDictionary(kv => kv.Value, kv => kv.Key);
                string Name(int g) => geomName.TryGetValue(g, out var n) ? n : $"geom{g}";
                var athletes = Athletes(m);
                Assert.GreaterOrEqual(athletes.Select(a => a.lane).Distinct().Count(), 8, "8 lanes of athletes");
                var excluded = new HashSet<(int, int)>();
                for (int i = 0; i < (int)m->nexclude; i++)
                {
                    int sig = m->exclude_signature[i];
                    excluded.Add((Math.Min(sig >> 16, sig & 0xFFFF), Math.Max(sig >> 16, sig & 0xFFFF)));
                }
                bool Passes(int a, int b)
                {
                    bool bits = (m->geom_contype[a] & m->geom_conaffinity[b]) != 0 || (m->geom_contype[b] & m->geom_conaffinity[a]) != 0;
                    int ba = m->geom_bodyid[a], bb = m->geom_bodyid[b];
                    return bits && !excluded.Contains((Math.Min(ba, bb), Math.Max(ba, bb)));
                }

                var key = new double[(int)m->nq];                    // the scene's compiled start state (athletes on their spots)
                for (int i = 0; i < (int)m->nq; i++) key[i] = d->qpos[i];
                MujocoLib.mj_kinematics(m, d);
                var baseXpos = new double[3 * (int)m->ngeom];
                for (int i = 0; i < baseXpos.Length; i++) baseXpos[i] = d->geom_xpos[i];

                var failures = new List<string>();
                int pairs = 0;
                foreach (var A in athletes)
                    foreach (var B in athletes)
                    {
                        if (A.lane == B.lane) continue;          // roster: the other body of a lane is switched off in play
                        foreach (var b in B.geoms)
                            foreach (var a in A.geoms)
                            {
                                pairs++;
                                if (!Passes(a, b)) { failures.Add($"filtered: {Name(a)} / {Name(b)}"); continue; }
                                for (int i = 0; i < (int)m->nq; i++) d->qpos[i] = key[i];
                                for (int k = 0; k < 3; k++)            // move B so part b overlaps part a (+ a small nudge)
                                    d->qpos[B.rootQpos + k] += baseXpos[3 * a + k] - baseXpos[3 * b + k] + Nudge[k];
                                MujocoLib.mj_kinematics(m, d);
                                MujocoLib.mj_collision(m, d);
                                bool hit = false;
                                for (int c = 0; c < (int)d->ncon && !hit; c++)
                                {
                                    var con = d->contact[c];
                                    hit = (con.geom1 == a && con.geom2 == b) || (con.geom1 == b && con.geom2 == a);
                                }
                                if (!hit) failures.Add($"no contact: {Name(a)} / {Name(b)}");
                            }
                    }
                Debug.Log($"[CrowdContacts] {scenePath}: {athletes.Count} athletes, {pairs} cross-lane part pairs, {failures.Count} failures");
                Assert.IsEmpty(failures.Take(20), $"{failures.Count} failures in {scenePath}");
                Assert.GreaterOrEqual(pairs, 8 * 7 * 17 * 17);
            }
            finally
            {
                scene.DestroyScene();
                UnityEngine.Object.DestroyImmediate(scene.gameObject);
            }
        }

        static readonly double[] Nudge = { 0.011, 0.007, 0.004 };   // = check_crowd_contacts.NUDGE

        sealed class Athlete { public string prefix; public int lane, rootQpos; public List<int> geoms = new(); }

        /// <summary>Every free-floating pelvis = one athlete (prefix L&lt;k&gt;_ / Z&lt;k&gt;_, lane k); its geoms = all geoms under it.</summary>
        static List<Athlete> Athletes(MujocoLib.mjModel_* m)
        {
            var list = new List<Athlete>();
            foreach (var (name, b) in AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody).Select(kv => (kv.Key, kv.Value)))
            {
                if (!name.EndsWith("pelvis") || m->body_parentid[b] != 0) continue;
                var p = name.Substring(0, name.Length - "pelvis".Length);
                var a = new Athlete { prefix = p, lane = int.Parse(p.Substring(1, p.Length - 2)), rootQpos = m->jnt_qposadr[m->body_jntadr[b]] };
                for (int g = 0; g < (int)m->ngeom; g++)
                    if (m->body_rootid[m->geom_bodyid[g]] == b) a.geoms.Add(g);
                list.Add(a);
            }
            return list;
        }

    }
}
