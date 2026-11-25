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
            n.inputs['Roughness'].default_value = max(0.6, n.inputs['Roughness'].default_value)

# ---------- pivot empty ----------
piv = bpy.data.objects.new('PIV', None); scn.collection.objects.link(piv)
arm.parent = piv
SCALE = 0.34 if KIND == 'dog' else 1.0
piv.scale = (SCALE,) * 3
vl.update()

# ---------- parts ----------
def own_part(n):
    if KIND == 'human':
        if n in ('Neck', 'Head') or n.startswith('Head'): return 'head'
        side = 'L' if n.endswith('.L') else ('R' if n.endswith('.R') else None)
        if side and any(k in n for k in ['UpperArm', 'LowerArm', 'Hand', 'Index', 'Middle', 'Ring', 'Pinky', 'Thumb']): return 'arm' + side
        if side and any(k in n for k in ['UpperLeg', 'LowerLeg', 'Foot', 'Toe']): return 'leg' + side
        return None
    else:
        if n.startswith('Neck') or n.startswith('Head') or n.startswith('Ear') or 'Jaw' in n or 'Eye' in n: return 'head'
        if n.startswith('Tail'): return 'tail'
        side = 'L' if n.endswith('.L') else ('R' if n.endswith('.R') else None)
        if side and n.startswith('Front') and ('Leg' in n or 'Foot' in n or 'Paw' in n): return 'legF' + side
        if side and n.startswith('Back') and ('Leg' in n or 'Foot' in n or 'Paw' in n): return 'legB' + side
        return None
def part_of(bname):
    b = arm.data.bones.get(bname)
    while b:
        p = own_part(b.name)
        if p: return p
        b = b.parent
    return 'torso' if KIND == 'human' else 'body'

PARTS = ['legL', 'legR', 'torso', 'armL', 'armR', 'head'] if KIND == 'human' else ['body', 'head', 'legFL', 'legFR', 'legBL', 'legBR', 'tail']

import bmesh
part_objs = {p: [] for p in PARTS}
for o in meshes:
    gnames = {g.index: g.name for g in o.vertex_groups}
    vpart = []
    for v in o.data.vertices:
        best, bw = None, -1
        for g in v.groups:
            if g.weight > bw and g.group in gnames: best, bw = gnames[g.group], g.weight
        vpart.append(part_of(best) if best else ('torso' if KIND == 'human' else 'body'))
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

# ---------- posing helpers (model space, model faces -Y) ----------
def reset_pose():
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
    vl.update()
def aim(name, d):
    pb = arm.pose.bones.get(name)
    if not pb: return
    vl.update()
    mw = arm.matrix_world
    head = mw @ pb.head; tail = mw @ pb.tail
    cur = (tail - head).normalized()
    q = cur.rotation_difference(Vector(d).normalized())
    R = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head)
    W = R @ (mw @ pb.matrix)
    pb.matrix = mw.inverted() @ W
    vl.update()
def rifle_arms():
    aim('UpperArm.R', (-0.18, -0.30, -0.94)); aim('LowerArm.R', (0.18, -0.97, -0.12)); aim('Hand.R', (0.1, -1, -0.05))
    aim('UpperArm.L', (0.22, -0.42, -0.88)); aim('LowerArm.L', (-0.55, -0.82, 0.12)); aim('Hand.L', (-0.4, -0.9, 0.05))
def legs(aL, kL, aR, kR):
    for side, a_, k in (('L', aL, kL), ('R', aR, kR)):
        aim('UpperLeg.' + side, (0, -math.sin(a_), -math.cos(a_)))
        aim('LowerLeg.' + side, (0, -math.sin(a_ - k), -math.cos(a_ - k)))
        aim('Foot.' + side, (0, -math.cos(a_ - k * .3) * 1.0, -0.25))
WALK = [(0.40, 0.05, -0.35, 0.35), (0.0, 0.05, 0.15, 0.75), (-0.35, 0.35, 0.40, 0.05), (0.15, 0.75, 0.0, 0.05)]

gun = None
def make_gun():
    global gun
    vl.update()
    pb = arm.pose.bones['Hand.R']; mw = arm.matrix_world
    h = mw @ pb.head
    bm = bmesh.new()
    def box(cx, cy, cz, sx, sy, sz):
        r = bmesh.ops.create_cube(bm, size=1)
        for v in r['verts']: v.co = Vector((cx + v.co.x * sx, cy + v.co.y * sy, cz + v.co.z * sz))
    box(0, 0.10, 0.02, 0.05, 0.22, 0.09)   # stock
    box(0, -0.12, 0.03, 0.06, 0.26, 0.08)  # receiver
    box(0, -0.42, 0.05, 0.03, 0.40, 0.03)  # barrel
    box(0, -0.10, -0.08, 0.04, 0.07, 0.14) # magazine
    me = bpy.data.meshes.new('gun'); bm.to_mesh(me); bm.free()
    mat = bpy.data.materials.new('GunMat'); mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.05, 0.05, 0.055, 1)
    mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.6
    mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.45
    me.materials.append(mat)
    gun = bpy.data.objects.new('armR__gun', me); scn.collection.objects.link(gun)
    W = Matrix.Translation(h + Vector((0.02, -0.05, -0.02)))
    gun.parent = arm; gun.parent_type = 'BONE'; gun.parent_bone = 'Hand.R'
    vl.update(); gun.matrix_world = W; vl.update()
    part_objs['armR'].append(gun)

def set_pose(pose):
    """pose: 0 idle, 1-4 walk, 5 fall, 6 dead. returns fall angle"""
    if KIND == 'human':
        reset_pose(); rifle_arms()
        if 1 <= pose <= 4: legs(*WALK[pose - 1])
        if pose == 5: legs(0.1, 0.3, -0.1, 0.2)
        if pose == 6: aim('UpperArm.L', (0.9, 0.2, -0.2)); aim('LowerArm.L', (1, 0, -0.1)); aim('UpperArm.R', (-0.9, -0.3, -0.2))
        return {5: 0.8, 6: 1.5708}.get(pose, 0.0)
    else:
        acts = {x.name.split('|')[-1]: x for x in bpy.data.actions}
        if not arm.animation_data: arm.animation_data_create()
        def use(name, t):
            act = acts[name]; arm.animation_data.action = act
            try:
                slot = act.slots[0] if hasattr(act, 'slots') and len(act.slots) else None
                if slot: arm.animation_data.action_slot = slot
            except Exception: pass
            f0, f1 = act.frame_range
            scn.frame_set(int(round(f0 + (f1 - f0) * t))); vl.update()
        if pose == 0: use('Idle', 0)
        elif pose <= 4: use('Walk', (pose - 1) / 4)
        elif pose == 5: use('Death', 0.45)
        else: use('Death', 1.0)
        return 0.0

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
    vl.material_override = None; scn.cycles.samples = 28; scn.cycles.filter_width = 1.2
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
    """joint positions in engine-local coords (forward +y), current pose & piv transform w/o yaw"""
    vl.update(); mw = arm.matrix_world; J = {}
    names = {'human': {'head': 'Neck', 'armL': 'UpperArm.L', 'armR': 'UpperArm.R', 'legL': 'UpperLeg.L', 'legR': 'UpperLeg.R', 'torsoC': 'Chest'},
             'dog': {'head': 'Neck1', 'legFL': 'FrontUpperLeg.L', 'legFR': 'FrontUpperLeg.R', 'legBL': 'BackUpperLeg.L', 'legBR': 'BackUpperLeg.R', 'tail': 'Tail1', 'torsoC': 'Torso2'}}[KIND]
    for k, bn in names.items():
        pb = arm.pose.bones.get(bn)
        if not pb: continue
        w = mw @ pb.head
        J[k] = [round(w.y * U, 2), round(w.x * U, 2), round(w.z * U * KZ, 2)]
    if gun is not None:
        g = gun.matrix_world @ Vector((0, -0.62, 0.05))
        J['muzzle'] = [round(g.y * U, 2), round(g.x * U, 2), round(g.z * U * KZ, 2)]
    return J

if KIND == 'human':
    reset_pose(); rifle_arms(); make_gun()

meta = {'parts': PARTS, 'poses': {}}
poses = [0] if MODE == 'preview' else list(range(7))
dirs = [7, 1] if MODE == 'preview' else list(range(8))
t0 = time.time()
for pose in poses:
    piv.matrix_world = Matrix.Diagonal((SCALE, SCALE, SCALE, 1)); vl.update()
    fall = set_pose(pose)
    piv.rotation_mode = 'XYZ'; piv.rotation_euler = (0, 0, 0); vl.update(); piv.matrix_world = Matrix.Rotation(math.pi / 2, 4, 'Z') @ Matrix.Rotation(-fall, 4, 'X') @ Matrix.Diagonal((SCALE, SCALE, SCALE, 1)); piv.location = (0, 0, 0); vl.update()
    lift = -min_z(); piv.location = (0, 0, lift); vl.update()
    meta['poses'][pose] = {'joints': joints_engine()}
    for d in dirs:
        piv.rotation_euler = (0, 0, 0); vl.update()
        piv.matrix_world = Matrix.Translation((0, 0, lift)) @ Matrix.Rotation(math.pi / 2 - d * math.pi / 4, 4, 'Z') @ Matrix.Rotation(-fall, 4, 'X') @ Matrix.Diagonal((SCALE, SCALE, SCALE, 1))
        vl.update()
        if MODE == 'preview':
            color_mode()
            for p in PARTS:
                for o in part_objs[p]: o.hide_render = False
            scn.render.filepath = f'{OUT}/prev_{d}.png'; bpy.ops.render.render(write_still=True)
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
