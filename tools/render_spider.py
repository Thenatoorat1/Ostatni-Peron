import bpy, sys, math, json, os, time
from mathutils import Vector, Matrix, Quaternion

a = sys.argv[sys.argv.index('--') + 1:]
KIND, SRC, OUT, MODE = a[0], a[1], a[2], a[3]          # KIND human|dog ; MODE preview|full
TINT = json.loads(a[4]) if len(a) > 4 else {}
U = 10.0; KZ = 2.449489743 / 2; P = 2 * math.sqrt(2) * U
RES = 128; AX, AY = 64, 100
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o.name.startswith('Icosphere'):
        bpy.data.objects.remove(o)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
scn = bpy.context.scene
vl = bpy.context.view_layer

# ---------- tint materials ----------
for m in bpy.data.materials:
    if not m.use_nodes: continue
    for n in m.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            c = n.inputs['Base Color'].default_value
            t = TINT.get(m.name)
            if t: n.inputs['Base Color'].default_value = (t[0], t[1], t[2], 1)
            n.inputs['Roughness'].default_value = 0.5; n.inputs['Metallic'].default_value = 0.0

# ---------- pivot empty ----------
piv = bpy.data.objects.new('PIV', None); scn.collection.objects.link(piv)
arm.parent = piv
SCALE = 0.3
piv.scale = (SCALE,) * 3
vl.update()

# ---------- parts ----------
def own_part(n):
    if n == 'Head': return 'head'
    if n == 'Abdomen': return 'abd'
    if 'Leg' in n and n.endswith('.L'): return 'legL'
    if 'Leg' in n and n.endswith('.R'): return 'legR'
    return None
def part_of(bname):
    b = arm.data.bones.get(bname)
    while b:
        p = own_part(b.name)
        if p: return p
        b = b.parent
    return 'body'
PARTS = ['body', 'head', 'abd', 'legL', 'legR']
import bmesh
part_objs = {p: [] for p in PARTS}
for o in meshes:
    gnames = {g.index: g.name for g in o.vertex_groups}
    vpart = []
    for v in o.data.vertices:
        best, bw = None, -1
        for g in v.groups:
            if g.weight > bw and g.group in gnames: best, bw = gnames[g.group], g.weight
        vpart.append(part_of(best) if best else 'body')
    fpart = []
    for f in o.data.polygons:
        cnt = {}
        for vi in f.vertices: cnt[vpart[vi]] = cnt.get(vpart[vi], 0) + 1
        fpart.append(max(cnt, key=cnt.get))
    for p in set(fpart):
        no = o.copy(); no.data = o.data.copy(); no.name = p + '__' + o.name
        for c in o.users_collection: c.objects.link(no)
        bm = bmesh.new(); bm.from_mesh(no.data); bm.faces.ensure_lookup_table()
        kill = [bm.faces[i] for i, fp in enumerate(fpart) if fp != p]
        bmesh.ops.delete(bm, geom=kill, context='FACES')
        bm.to_mesh(no.data); bm.free()
        part_objs[p].append(no)
    bpy.data.objects.remove(o)

# ---------- poses from the model's own actions ----------
ACT = {a.name: a for a in bpy.data.actions}
print('ACTIONS', list(ACT))
POSES = ['idle1', 'idle2', 'walk1', 'walk2', 'walk3', 'walk4', 'bite1', 'bite2', 'spit', 'jump1', 'jump2', 'd1', 'd2', 'd3', 'wu1', 'wu2', 'wu3', 'wu4', 'wd1', 'wd2', 'wd3', 'wd4']
PF = {'idle1': ('Idle', 0), 'idle2': ('Idle', 50), 'walk1': ('Walk', 0), 'walk2': ('Walk', 5), 'walk3': ('Walk', 10), 'walk4': ('Walk', 15),
      'bite1': ('Attack', 5), 'bite2': ('Attack', 10), 'spit': ('Attack', 2), 'jump1': ('Jump', 4), 'jump2': ('Jump', 9),
      'd1': ('Death', 6), 'd2': ('Death', 14), 'd3': ('Death', 25),
      'wu1': ('Walk', 0), 'wu2': ('Walk', 5), 'wu3': ('Walk', 10), 'wu4': ('Walk', 15), 'wd1': ('Walk', 0), 'wd2': ('Walk', 5), 'wd3': ('Walk', 10), 'wd4': ('Walk', 15)}
if not arm.animation_data: arm.animation_data_create()
def set_pose(pi):
    an, f = PF[POSES[pi]]
    act = [a for n, a in ACT.items() if n.endswith(an)][0]
    arm.animation_data.action = act
    try:
        if act.slots: arm.animation_data.action_slot = act.slots[0]
    except Exception: pass
    scn.frame_set(int(f)); vl.update()
    pb = arm.pose.bones['Abdomen']; pb.scale = (1.35, 1.45, 1.3)
    hb = arm.pose.bones['Head']; hb.scale = (1.15, 1.15, 1.15); vl.update()
    n = POSES[pi]
    # on a wall: belly towards the facing direction (the wall), head up (wu) or down (wd)
    if n.startswith('wu'): return Matrix.Rotation(-math.pi / 2, 4, 'X')
    if n.startswith('wd'): return Matrix.Rotation(math.pi, 4, 'Z') @ Matrix.Rotation(math.pi / 2, 4, 'X')
    return Matrix.Identity(4)
gun = None
# ---------- scene, camera, light ----------
scn.render.engine = 'CYCLES'
scn.cycles.device = 'CPU'
scn.cycles.use_denoising = False
scn.render.film_transparent = True
scn.render.resolution_x = scn.render.resolution_y = RES
scn.render.resolution_percentage = 100
scn.render.use_persistent_data = True
scn.render.image_settings.file_format = 'PNG'
scn.render.image_settings.color_mode = 'RGBA'
scn.render.image_settings.color_depth = '16'
try: scn.view_settings.view_transform = 'Standard'
except Exception as e: print('VT', e)
world = bpy.data.worlds.new('W'); scn.world = world; world.use_nodes = True
bgn = world.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.62, 0.66, 0.74, 1); bgn.inputs['Strength'].default_value = 1.0
sun = bpy.data.lights.new('Sun', 'SUN'); sun.energy = 4.6; sun.angle = math.radians(4)
so = bpy.data.objects.new('Sun', sun); scn.collection.objects.link(so)
Ld = Vector((0.3, -0.6, 1)).normalized()  # engine L (-0.6,0.3,1) with x/y swapped
so.rotation_mode = 'QUATERNION'; so.rotation_quaternion = (-Ld).to_track_quat('-Z', 'Y')
cam = bpy.data.cameras.new('Cam'); cam.type = 'ORTHO'; cam.ortho_scale = RES / P
co = bpy.data.objects.new('Cam', cam); scn.collection.objects.link(co); scn.camera = co
e = math.radians(30)
Dc = Vector((math.cos(e) * math.cos(math.radians(45)), math.cos(e) * math.sin(math.radians(45)), math.sin(e)))
Up = Vector((-math.sin(e) * math.cos(math.radians(45)), -math.sin(e) * math.sin(math.radians(45)), math.cos(e)))
C = Up * ((AY - RES / 2) / P)
co.location = C + Dc * 30
co.rotation_mode = 'QUATERNION'; co.rotation_quaternion = (-Dc).to_track_quat('-Z', 'Y')
cam.clip_end = 100

# position material
pm = bpy.data.materials.new('POS'); pm.use_nodes = True
nt = pm.node_tree; nt.nodes.clear()
geo = nt.nodes.new('ShaderNodeNewGeometry'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sep.inputs[0])
def math_(op, a_, b_):
    n = nt.nodes.new('ShaderNodeMath'); n.operation = op
    for i, v in enumerate((a_, b_)):
        if isinstance(v, (int, float)): n.inputs[i].default_value = v
        else: nt.links.new(v, n.inputs[i])
    return n.outputs[0]
xy = math_('ADD', sep.outputs[0], sep.outputs[1])
dep = math_('ADD', math_('MULTIPLY', xy, U), math_('MULTIPLY', sep.outputs[2], U * KZ))
R = math_('MULTIPLY', math_('ADD', dep, 64), 1 / 128)
Gc = math_('MULTIPLY', sep.outputs[2], U * KZ / 64)
cc = nt.nodes.new('ShaderNodeCombineColor'); nt.links.new(R, cc.inputs[0]); nt.links.new(Gc, cc.inputs[1]); cc.inputs[2].default_value = 0.5
em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(cc.outputs[0], em.inputs[0]); em.inputs[1].default_value = 1
out = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(em.outputs[0], out.inputs[0])

def color_mode():
    vl.material_override = None; scn.cycles.samples = 20; scn.cycles.filter_width = 1.2
    scn.cycles.max_bounces = 3
def pos_mode():
    vl.material_override = pm; scn.cycles.samples = 1; scn.cycles.filter_width = 0.01; scn.cycles.max_bounces = 0

def min_z():
    dg = bpy.context.evaluated_depsgraph_get(); mz = 1e9
    for p in PARTS:
        for o in part_objs[p]:
            ev = o.evaluated_get(dg); me = ev.to_mesh()
            mw = ev.matrix_world
            for v in me.vertices:
                z = (mw @ v.co).z
                if z < mz: mz = z
            ev.to_mesh_clear()
    return mz

def joints_engine():
    vl.update(); mw = arm.matrix_world; J = {}
    for k, bn in {'head': 'Head', 'abd': 'Abdomen', 'legL': 'MidFrontLeg.L', 'legR': 'MidFrontLeg.R', 'torsoC': 'Body'}.items():
        pb = arm.pose.bones.get(bn); w = mw @ pb.head
        J[k] = [round(w.y * U, 2), round(w.x * U, 2), round(w.z * U * KZ, 2)]
    return J
meta = {'parts': PARTS, 'poses': {}, 'poseNames': POSES}
poses = [POSES.index(x) for x in (sys.argv[-1].split(',') if MODE == 'preview' else POSES)] if MODE == 'preview' else list(range(len(POSES)))
dirs = [7, 1] if MODE == 'preview' else list(range(8))
t0 = time.time()
for pose in poses:
    piv.matrix_world = Matrix.Diagonal((SCALE, SCALE, SCALE, 1)); vl.update()
    fall = set_pose(pose)
    piv.rotation_mode = 'XYZ'; piv.rotation_euler = (0, 0, 0); vl.update(); piv.matrix_world = Matrix.Rotation(math.pi / 2, 4, 'Z') @ fall @ Matrix.Diagonal((SCALE, SCALE, SCALE, 1)); piv.location = (0, 0, 0); vl.update()
    lift = -min_z(); piv.location = (0, 0, lift); vl.update()
    meta['poses'][pose] = {'joints': joints_engine()}
    for d in dirs:
        piv.rotation_euler = (0, 0, 0); vl.update()
        piv.matrix_world = Matrix.Translation((0, 0, lift)) @ Matrix.Rotation(math.pi / 2 - d * math.pi / 4, 4, 'Z') @ fall @ Matrix.Diagonal((SCALE, SCALE, SCALE, 1))
        vl.update()
        if MODE == 'preview':
            color_mode()
            for p in PARTS:
                for o in part_objs[p]: o.hide_render = False
            scn.render.filepath = f'{OUT}/prev_{POSES[pose]}_{d}.png'; bpy.ops.render.render(write_still=True)
            continue
        for mode in ('c', 'p'):
            color_mode() if mode == 'c' else pos_mode()
            for p in PARTS:
                for q in PARTS:
                    for o in part_objs[q]: o.hide_render = (q != p)
                scn.render.filepath = f'{OUT}/{mode}_{pose}_{d}_{p}.png'
                bpy.ops.render.render(write_still=True)
    print('pose', pose, 'done', round(time.time() - t0, 1), flush=True)
json.dump(meta, open(f'{OUT}/meta.json', 'w'))
print('ALLDONE', flush=True)
