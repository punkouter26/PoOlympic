using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Mujoco;

namespace PoOlympic
{
    /// <summary>
    /// Parity gate G0 — dumps the physics-relevant fields of the compiled mjModel in exactly the schema of
    /// training/poolympic/fingerprint.py, so the two can be compared field by field
    /// (`uv run python -m poolympic.fingerprint parity/fingerprint_python.json parity/fingerprint_unity.json`).
    /// </summary>
    public static unsafe class ModelFingerprint
    {
        const int ObjBody = (int)MujocoLib.mjtObj.mjOBJ_BODY;
        const int ObjJoint = (int)MujocoLib.mjtObj.mjOBJ_JOINT;
        const int ObjGeom = (int)MujocoLib.mjtObj.mjOBJ_GEOM;
        const int ObjActuator = (int)MujocoLib.mjtObj.mjOBJ_ACTUATOR;
        const int HingeType = 3; // mjJNT_HINGE

        static readonly System.Text.RegularExpressions.Regex PluginSuffix = new(@"_\d+$");

        /// <summary>
        /// Element name with the Unity plug-in's uniqueness suffix removed: the plug-in names every MuJoCo element
        /// "&lt;GameObject name&gt;_&lt;n&gt;". Generated MJCF names never end in "_&lt;digits&gt;" (build_mjcf.py).
        /// </summary>
        public static string Name(MujocoLib.mjModel_* m, int type, int id)
        {
            var p = MujocoLib.mj_id2name(m, type, id);
            return p == IntPtr.Zero ? "" : PluginSuffix.Replace(Marshal.PtrToStringAnsi(p), "");
        }

        /// <summary>mj_name2id for a body or joint by its MJCF name (without the plug-in suffix); -1 if absent. A raw
        /// mj_name2id("shaker_x") misses "shaker_x_255" (Gust Gauntlet threw on every shake round in Unity).</summary>
        public static int Id(MujocoLib.mjModel_* m, int type, string name)
        {
            int id = MujocoLib.mj_name2id(m, type, name);
            if (id >= 0) return id;
            int count = type == ObjBody ? (int)m->nbody : type == ObjJoint ? (int)m->njnt : type == ObjGeom ? (int)m->ngeom : 0;
            for (int i = 0; i < count; i++)
                if (Name(m, type, i) == name) return i;
            return -1;
        }

        public static string Dump(MujocoLib.mjModel_* m, string athletePrefix = "", string cubeName = "cube0")
        {
            string Keep(string n) => athletePrefix.Length == 0 ? n : (n.StartsWith(athletePrefix) ? n.Substring(athletePrefix.Length) : null);

            var j = new Json();
            j.Open();
            j.Key("schema").Int(1);

            var o = m->opt;
            j.Key("option").Open();
            j.Key("timestep").Num(o.timestep);
            j.Key("gravity").Arr(o.gravity, 3);
            j.Key("integrator").Int(o.integrator);
            j.Key("cone").Int(o.cone);
            j.Key("jacobian").Int(o.jacobian);
            j.Key("solver").Int(o.solver);
            j.Key("iterations").Int(o.iterations);
            j.Key("tolerance").Num(o.tolerance);
            j.Key("ls_iterations").Int(o.ls_iterations);
            j.Key("ls_tolerance").Num(o.ls_tolerance);
            j.Key("impratio").Num(o.impratio);
            j.Key("disableflags").Int(o.disableflags);
            j.Key("enableflags").Int(o.enableflags);
            j.Close();

            int nbody = (int)m->nbody, njnt = (int)m->njnt, ngeom = (int)m->ngeom, nu = (int)m->nu;
            int athleteBodies = 0;

            j.Key("bodies").Open();
            for (int b = 1; b < nbody; b++)
            {
                var name = Name(m, ObjBody, b);
                bool isCube = name == cubeName;
                var key = isCube ? "cube" : Keep(name);
                if (key == null || (name.StartsWith("cube") && !isCube)) continue;
                if (!isCube) athleteBodies++;
                int pid = m->body_parentid[b];
                var parent = pid == 0 ? "world" : (Keep(Name(m, ObjBody, pid)) ?? Name(m, ObjBody, pid));
                j.Key(key).Open();
                j.Key("parent").Str(parent);
                j.Key("pos"); if (isCube) j.Null(); else j.Arr(m->body_pos + 3 * b, 3);
                j.Key("quat").Arr(m->body_quat + 4 * b, 4);
                j.Key("mass").Num(m->body_mass[b]);
                j.Key("inertia").Arr(m->body_inertia + 3 * b, 3);
                j.Key("ipos").Arr(m->body_ipos + 3 * b, 3);
                j.Key("iquat").Arr(m->body_iquat + 4 * b, 4);
                j.Key("gravcomp").Num(m->body_gravcomp[b]);
                j.Close();
            }
            j.Close();

            int athleteJoints = 0;
            j.Key("joints").Open();
            for (int i = 0; i < njnt; i++)
            {
                var body = Name(m, ObjBody, m->jnt_bodyid[i]);
                if (body.StartsWith("cube")) continue;
                var key = Keep(Name(m, ObjJoint, i));
                if (key == null) continue;
                athleteJoints++;
                int d = m->jnt_dofadr[i];
                j.Key(key).Open();
                j.Key("type").Int(m->jnt_type[i]);
                j.Key("body").Str(Keep(body));
                j.Key("pos").Arr(m->jnt_pos + 3 * i, 3);
                j.Key("axis").Arr(m->jnt_axis + 3 * i, 3);
                j.Key("limited").Int(m->jnt_limited[i]);
                j.Key("range").Arr(m->jnt_range + 2 * i, 2);
                j.Key("stiffness").Num(m->jnt_stiffness[i]);
                j.Key("springref"); if (m->jnt_type[i] == HingeType) j.Num(m->qpos_spring[m->jnt_qposadr[i]]); else j.Null();
                j.Key("armature").Num(m->dof_armature[d]);
                j.Key("damping").Num(m->dof_damping[d]);
                j.Key("frictionloss").Num(m->dof_frictionloss[d]);
                j.Key("solref").Arr(m->jnt_solref + MujocoLib.mjNREF * i, MujocoLib.mjNREF);
                j.Key("solimp").Arr(m->jnt_solimp + MujocoLib.mjNIMP * i, MujocoLib.mjNIMP);
                j.Close();
            }
            j.Close();

            j.Key("geoms").Open();
            for (int g = 0; g < ngeom; g++)
            {
                int bid = m->geom_bodyid[g];
                var body = bid == 0 ? "world" : Name(m, ObjBody, bid);
                var name = Name(m, ObjGeom, g);
                string key;
                if (name == "ground") key = "ground";
                else if (body == cubeName) key = "cube_geom";
                else if (body.StartsWith("cube") || body == "world") continue;
                else { key = Keep(name); if (key == null) continue; }
                j.Key(key).Open();
                j.Key("type").Int(m->geom_type[g]);
                j.Key("body").Str(body == "world" ? "world" : (Keep(body) ?? body));
                j.Key("size"); if (name == "ground") j.Null(); else j.Arr(m->geom_size + 3 * g, 3);
                j.Key("pos").Arr(m->geom_pos + 3 * g, 3);
                j.Key("quat").Arr(m->geom_quat + 4 * g, 4);
                j.Key("contype").Int(m->geom_contype[g]);
                j.Key("conaffinity").Int(m->geom_conaffinity[g]);
                j.Key("condim").Int(m->geom_condim[g]);
                j.Key("friction").Arr(m->geom_friction + 3 * g, 3);
                j.Key("solref").Arr(m->geom_solref + MujocoLib.mjNREF * g, MujocoLib.mjNREF);
                j.Key("solimp").Arr(m->geom_solimp + MujocoLib.mjNIMP * g, MujocoLib.mjNIMP);
                j.Key("solmix").Num(m->geom_solmix[g]);
                j.Key("margin").Num(m->geom_margin[g]);
                j.Key("gap").Num(m->geom_gap[g]);
                j.Key("priority").Int(m->geom_priority[g]);
                j.Close();
            }
            j.Close();

            int athleteActuators = 0;
            j.Key("actuators").Open();
            for (int a = 0; a < nu; a++)
            {
                var key = Keep(Name(m, ObjActuator, a));
                if (key == null) continue;
                athleteActuators++;
                j.Key(key).Open();
                j.Key("joint").Str(Keep(Name(m, ObjJoint, m->actuator_trnid[2 * a])));
                j.Key("dyntype").Int(m->actuator_dyntype[a]);
                j.Key("gaintype").Int(m->actuator_gaintype[a]);
                j.Key("biastype").Int(m->actuator_biastype[a]);
                j.Key("gainprm").Arr(m->actuator_gainprm + MujocoLib.mjNGAIN * a, 3);
                j.Key("biasprm").Arr(m->actuator_biasprm + MujocoLib.mjNBIAS * a, 3);
                j.Key("gear").Num(m->actuator_gear[6 * a]);
                j.Key("ctrllimited").Int(m->actuator_ctrllimited[a]);
                j.Key("forcelimited").Int(m->actuator_forcelimited[a]);
                j.Key("forcerange").Arr(m->actuator_forcerange + 2 * a, 2);
                j.Close();
            }
            j.Close();

            j.Key("counts").Open();
            j.Key("athlete_bodies").Int(athleteBodies);
            j.Key("athlete_joints").Int(athleteJoints);
            j.Key("athlete_actuators").Int(athleteActuators);
            j.Close();

            // <contact><exclude> body pairs of this athlete: "a|b,c|d", names sorted. Not part of the Python fingerprint
            // hash (schema 1 predates it; adding it would re-key every brain) — G0 compares it separately
            // (training/poolympic/fingerprint.py excludes()).
            var excludes = new System.Collections.Generic.List<string>();
            for (int i = 0; i < (int)m->nexclude; i++)
            {
                int sig = m->exclude_signature[i];
                string a = Keep(Name(m, ObjBody, sig >> 16)), b = Keep(Name(m, ObjBody, sig & 0xFFFF));
                if (a == null || b == null) continue;
                excludes.Add(string.CompareOrdinal(a, b) <= 0 ? a + "|" + b : b + "|" + a);
            }
            excludes.Sort(string.CompareOrdinal);
            j.Key("excludes").Str(string.Join(",", excludes));

            j.Close();
            return j.ToString();
        }

        /// <summary>Minimal JSON writer (round-trip doubles, no dependencies).</summary>
        sealed class Json
        {
            readonly StringBuilder _sb = new();
            readonly Stack<bool> _first = new();

            void Sep()
            {
                if (_first.Count == 0) return;
                if (!_first.Peek()) _sb.Append(',');
                _first.Pop();
                _first.Push(false);
            }

            public Json Open()
            {
                if (!_pendingValue) Sep();
                _pendingValue = false;
                _sb.Append('{');
                _first.Push(true);
                return this;
            }
            public Json Close() { _sb.Append('}'); _first.Pop(); return this; }
            public Json Key(string k) { Sep(); _sb.Append('"').Append(k).Append("\":"); _pendingValue = true; return this; }
            bool _pendingValue;

            void Val(string raw)
            {
                if (!_pendingValue) Sep();
                _pendingValue = false;
                _sb.Append(raw);
            }

            public Json Int(long v) { Val(v.ToString(CultureInfo.InvariantCulture)); return this; }
            public Json Num(double v) { Val(Fmt(v)); return this; }
            public Json Str(string s) { Val(s == null ? "null" : "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""); return this; }
            public Json Null() { Val("null"); return this; }

            public Json Arr(double* p, int n)
            {
                var parts = new string[n];
                for (int i = 0; i < n; i++) parts[i] = Fmt(p[i]);
                Val("[" + string.Join(",", parts) + "]");
                return this;
            }

            static string Fmt(double v)
            {
                var s = v.ToString("R", CultureInfo.InvariantCulture);
                return s.Contains(".") || s.Contains("E") || s.Contains("N") || s.Contains("I") ? s : s + ".0";
            }

            public override string ToString() => _sb.ToString();
        }
    }
}
