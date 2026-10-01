using System;
using System.Collections.Generic;
using Mujoco;

namespace PoOlympic
{
    /// <summary>
    /// Elimination rule for one athlete, read from mjData only (DESIGN §1 fall rule + leaving its support):
    /// pelvis below 0.55 m, torso tilt over 60°, a foot/toe geom centre below the support top ("stepped off"), or any
    /// non-foot body of this athlete touching its support surfaces. Mirrors training/poolympic/events/iron_pedestal.py.
    /// </summary>
    public sealed unsafe class AthleteJudge
    {
        public double fallPelvisZ = 0.55, fallTiltDeg = 60, steppedOffZ = -0.06;
        readonly PolicyRunner _runner;
        readonly int _torso, _pelvis;
        readonly int[] _feet;
        readonly HashSet<int> _support = new();

        /// <param name="supportGeoms">geom names this athlete may stand on (e.g. "ground", "L3_pedestal")</param>
        public AthleteJudge(MujocoLib.mjModel_* m, PolicyRunner runner, params string[] supportGeoms)
        {
            _runner = runner;
            fallPelvisZ = runner.Contract.FallPelvisZ;   // per body (MATT 0.55 m, zombie 0.324 m)
            var bodies = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_BODY, (int)m->nbody);
            var geoms = AthleteBinding.NameIndex(m, (int)MujocoLib.mjtObj.mjOBJ_GEOM, (int)m->ngeom);
            var p = runner.athletePrefix;
            _torso = bodies[p + "torso"];
            _pelvis = bodies[p + "pelvis"];
            _feet = new[] { geoms[p + "foot_l_geom0"], geoms[p + "toe_l_geom0"], geoms[p + "foot_r_geom0"], geoms[p + "toe_r_geom0"] };
            foreach (var n in supportGeoms)
                if (geoms.TryGetValue(n, out var g)) _support.Add(g);
        }

        /// <summary>Null while the athlete is still in; otherwise "FELL" or "STEPPED OFF".</summary>
        public string Eliminated(MujocoLib.mjModel_* m, MujocoLib.mjData_* d)
        {
            if (_runner.PelvisHeight(d) < fallPelvisZ) return "FELL";
            if (Math.Acos(Math.Clamp(d->xmat[9 * _torso + 8], -1, 1)) * 180 / Math.PI > fallTiltDeg) return "FELL";
            foreach (var g in _feet)
                if (d->geom_xpos[3 * g + 2] < steppedOffZ) return "STEPPED OFF";
            for (int i = 0; i < d->ncon; i++)
            {
                var c = d->contact[i];
                int other = _support.Contains(c.geom1) ? c.geom2 : _support.Contains(c.geom2) ? c.geom1 : -1;
                if (other < 0 || Array.IndexOf(_feet, other) >= 0) continue;
                if (m->body_rootid[m->geom_bodyid[other]] == _pelvis) return "FELL";
            }
            return null;
        }

        /// <summary>True while the foot or toe geom of one side (0 left, 1 right) touches a support surface
        /// (flamingo.py touchdown).</summary>
        public bool FootDown(MujocoLib.mjData_* d, int side)
        {
            int foot = _feet[2 * side], toe = _feet[2 * side + 1];
            for (int i = 0; i < d->ncon; i++)
            {
                var c = d->contact[i];
                int other = _support.Contains(c.geom1) ? c.geom2 : _support.Contains(c.geom2) ? c.geom1 : -1;
                if (other == foot || other == toe) return true;
            }
            return false;
        }

        public (double x, double y, double z) Pelvis(MujocoLib.mjData_* d) =>
            (d->xpos[3 * _pelvis], d->xpos[3 * _pelvis + 1], d->xpos[3 * _pelvis + 2]);
    }
}
