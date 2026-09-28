
import bpy, bmesh, math
from mathutils import Vector
L = 84.39          # straight length (m)
R0 = 36.5          # measurement-line radius of lane 1 kerb (m)
LANE_W = 1.22
N_LANES = 8
TRACK_IN = R0 - 0.30            # inner kerb edge
TRACK_OUT = R0 + N_LANES * LANE_W + 1.2   # outer apron of the track

def oval(r, n_arc=64):
    """Points on the stadium curve at distance r from the centre segment [-L/2, L/2] on x (counter-clockwise)."""
    pts = []
    for i in range(n_arc + 1):  # right bend
        a = -math.pi / 2 + math.pi * i / n_arc
        pts.append((L / 2 + r * math.cos(a), r * math.sin(a)))
    for i in range(n_arc + 1):  # left bend
        a = math.pi / 2 + math.pi * i / n_arc
        pts.append((-L / 2 + r * math.cos(a), r * math.sin(a)))
    return pts

def get_col(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        root = bpy.data.collections.get("Stadium") or bpy.data.collections.new("Stadium")
        if root.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(root)
        if col is not root:
            root.children.link(col)
    return col

def mesh_obj(name, verts, faces, mat=None, col="Stadium"):
    old = bpy.data.objects.get(name)
    if old: bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    get_col(col).objects.link(ob)
    if mat: me.materials.append(mat)
    return ob

def ring(name, r_in, r_out, z, mat, col="Stadium", n_arc=64):
    a, b = oval(r_in, n_arc), oval(r_out, n_arc)
    n = len(a)
    verts = [(x, y, z) for x, y in a] + [(x, y, z) for x, y in b]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return mesh_obj(name, verts, faces, mat, col)

def wall(name, r, z0, z1, mat, col="Stadium", n_arc=64):
    a = oval(r, n_arc)
    n = len(a)
    verts = [(x, y, z0) for x, y in a] + [(x, y, z1) for x, y in a]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return mesh_obj(name, verts, faces, mat, col)

def mat(name, color, rough=0.8, metal=0.0, emit=None, strength=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(nd for nd in m.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m
