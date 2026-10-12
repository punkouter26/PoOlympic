using System;
using Mujoco;
using UnityEngine;

namespace PoOlympic
{
    /// <summary>
    /// Collision-shape view (player option GameSettings.ColliderView): the MuJoCo collision primitives of every moving body
    /// (athletes, pool cubes), drawn through the skinned meshes straight from mjData (geom_xpos / geom_xmat and the
    /// model's geom sizes — the shapes the physics actually uses, not a copy of them). A geom turns from `rest` to
    /// `loaded` with the contact force on it; every contact above arrowMinN gets an arrow along its normal, as long as
    /// the force. Read only, drawn with Graphics.DrawMesh: no scene objects. During an instant replay the shapes
    /// follow the replayed pose, without forces (the replay records qpos, not contacts).
    /// </summary>
    [DefaultExecutionOrder(200)]
    public unsafe class ColliderXray : MonoBehaviour
    {
        [Header("Unity primitives (any size: scaled by their bounds)")]
        public Mesh sphere;
        public Mesh cylinder, cube;
        [Tooltip("PoOlympic/Xray: drawn over everything.")]
        public Material material;
        public Color rest = new(0.35f, 0.9f, 1f, 0.16f), loaded = new(1f, 0.25f, 0.15f, 0.75f), arrow = new(1f, 0.85f, 0.2f, 0.9f);
        [Tooltip("Contact force (N) at which a geom is fully `loaded`.")]
        public float fullForceN = 1500f;
        public float arrowMinN = 30f, arrowMetresPerKn = 0.5f, arrowMaxM = 1.5f;

        /// <summary>Shapes drawn last frame (tests, perf overlay).</summary>
        public int Drawn { get; private set; }

        const int Sphere = 2, Capsule = 3, Ellipsoid = 4, Cylinder = 5, Box = 6;       // mjtGeom
        static readonly int BaseColor = Shader.PropertyToID("_BaseColor");
        float[] _force;
        MaterialPropertyBlock _block;

        void LateUpdate()
        {
            Drawn = 0;
            if (!GameSettings.ColliderView || material == null || !MjScene.InstanceExists || MjScene.Instance.Model == null) return;
            var m = MjScene.Instance.Model;
            var d = MjScene.Instance.Data;
            int n = (int)m->ngeom;
            if (_force == null || _force.Length != n) _force = new float[n];
            else Array.Clear(_force, 0, n);
            _block ??= new MaterialPropertyBlock();

            if (Time.timeScale > 0.01f)             // frozen (replay, hit-stop): the contacts are not those of the pose shown
            {
                var f = stackalloc double[6];
                for (int i = 0; i < d->ncon; i++)
                {
                    var c = d->contact[i];
                    MujocoLib.mj_contactForce(m, d, i, f);
                    float fn = (float)Math.Abs(f[0]);
                    _force[c.geom1] += fn;
                    _force[c.geom2] += fn;
                    if (fn < arrowMinN) continue;
                    // the normal runs geom1 → geom2: point it at the moving body
                    var dir = new Vector3((float)c.frame[0], (float)c.frame[2], (float)c.frame[1]);
                    if (m->body_weldid[m->geom_bodyid[c.geom2]] == 0) dir = -dir;
                    float len = Mathf.Min(arrowMaxM, fn * 0.001f * arrowMetresPerKn);
                    var at = new Vector3((float)c.pos[0], (float)c.pos[2], (float)c.pos[1]);
                    Draw(cylinder, at + dir * (len * 0.5f), Quaternion.FromToRotation(Vector3.up, dir), new Vector3(0.012f, len * 0.5f, 0.012f), arrow);
                }
            }

            // ponytail: one draw call per shape (~500 with 8 athletes); instance them if this view ever has to run on a phone
            for (int g = 0; g < n; g++)
            {
                if (m->body_weldid[m->geom_bodyid[g]] == 0) continue;           // static: the scene shows it already
                double* s = m->geom_size + 3 * g, x = d->geom_xpos + 3 * g, r = d->geom_xmat + 9 * g;
                var pos = new Vector3((float)x[0], (float)x[2], (float)x[1]);
                // MuJoCo y ↔ Unity z, and a geom's long axis is its z (Unity primitives: y)
                var up = new Vector3((float)r[2], (float)r[8], (float)r[5]);
                var rot = Quaternion.LookRotation(new Vector3((float)r[1], (float)r[7], (float)r[4]), up);
                var color = Color.Lerp(rest, loaded, Mathf.Clamp01(_force[g] / fullForceN));
                float rad = (float)s[0], half = (float)s[1];
                switch (m->geom_type[g])
                {
                    case Sphere: Draw(sphere, pos, rot, Vector3.one * rad, color); break;
                    case Capsule:
                        Draw(cylinder, pos, rot, new Vector3(rad, half, rad), color);
                        Draw(sphere, pos + up * half, rot, Vector3.one * rad, color);
                        Draw(sphere, pos - up * half, rot, Vector3.one * rad, color);
                        break;
                    case Cylinder: Draw(cylinder, pos, rot, new Vector3(rad, half, rad), color); break;
                    case Ellipsoid: Draw(sphere, pos, rot, new Vector3(rad, (float)s[2], half), color); break;
                    case Box: Draw(cube, pos, rot, new Vector3(rad, (float)s[2], half), color); break;
                }
            }
        }

        /// <param name="half">half extents wanted along the mesh's own axes (the mesh may be any size: the built-in
        /// sphere and cylinder come at radius 1 here, not Unity's usual 0.5)</param>
        void Draw(Mesh mesh, Vector3 pos, Quaternion rot, Vector3 half, Color color)
        {
            if (mesh == null) return;
            var e = mesh.bounds.extents;
            var scale = new Vector3(half.x / e.x, half.y / e.y, half.z / e.z);
            _block.SetColor(BaseColor, color);
            Graphics.DrawMesh(mesh, Matrix4x4.TRS(pos, rot, scale), material, 0, null, 0, _block, false, false);
            Drawn++;
        }
    }
}
