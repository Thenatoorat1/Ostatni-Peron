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

import math as _m
def lean(a):
    aim('Torso', (0, -_m.sin(a * .55), _m.cos(a * .55))); aim('Chest', (0, -_m.sin(a), _m.cos(a)))
    aim('Head', (0, -_m.sin(a * .3) - .05, _m.cos(a * .3)))
def recoil_arms():
    aim('UpperArm.R', (-0.2, -0.12, -0.97)); aim('LowerArm.R', (0.18, -0.88, 0.38)); aim('Hand.R', (0.08, -0.92, 0.38))
    aim('UpperArm.L', (0.22, -0.3, -0.92)); aim('LowerArm.L', (-0.5, -0.78, 0.35)); aim('Hand.L', (-0.35, -0.85, 0.35))
def crouch_legs(ph=0.0):
    for side, sg in (('L', 1), ('R', -1)):
        a_ = 1.08 + 0.28 * ph * sg
        aim('UpperLeg.' + side, (sg * 0.13, -_m.sin(a_), -_m.cos(a_)))
        b_ = a_ - 2.25
        aim('LowerLeg.' + side, (0, -_m.sin(b_), -_m.cos(b_)))
        aim('Foot.' + side, (0, -1, -0.12))
def kneel_legs():
    for side, sg in (('L', 1), ('R', -1)):
        aim('UpperLeg.' + side, (sg * 0.12, -0.12, -0.99)); aim('LowerLeg.' + side, (0, 0.98, -0.15)); aim('Foot.' + side, (0, 0.5, -0.86))
def prone_arms(reach=0.0, recoil=False):
    aim('UpperArm.R', (-0.32, -0.5 + 0.15 * reach, 0.8)); aim('LowerArm.R', (0.35, -0.1, 0.93)); aim('Hand.R', (0.1, -0.25 if recoil else -0.1, 0.97))
    aim('UpperArm.L', (0.35, -0.55 - 0.15 * reach, 0.75)); aim('LowerArm.L', (-0.45, -0.1, 0.89)); aim('Hand.L', (-0.3, 0, 0.95))
    aim('Head', (0, 0.55, 0.83))
def prone_legs(ph=0.0):
    for side, sg in (('L', 1), ('R', -1)):
        if ph * sg > 0.5:
            aim('UpperLeg.' + side, (sg * 0.55, 0.15, -0.82)); aim('LowerLeg.' + side, (sg * 0.15, 0.4, -0.9))
        else:
            aim('UpperLeg.' + side, (sg * 0.15, 0.05, -0.99)); aim('LowerLeg.' + side, (sg * 0.05, 0.1, -0.99))
        aim('Foot.' + side, (0, 0.95, -0.3))
def dangle_arms():
    aim('UpperArm.R', (-0.25, -0.2, -0.95)); aim('LowerArm.R', (-0.05, -0.6, -0.8))
    aim('UpperArm.L', (0.25, -0.1, -0.96)); aim('LowerArm.L', (0.1, -0.3, -0.95))
def fling_arms():
    aim('UpperArm.R', (-0.75, 0.35, 0.55)); aim('LowerArm.R', (-0.4, 0.3, 0.86))
    aim('UpperArm.L', (0.75, 0.3, 0.6)); aim('LowerArm.L', (0.45, 0.2, 0.87))
RUN = [(0.75, 0.25, -0.55, 0.8), (0.1, 1.2, 0.25, 0.15), (-0.55, 0.8, 0.75, 0.25), (0.25, 0.15, 0.1, 1.2)]
def human_pose(name):
    reset_pose()
    if name == 'idle': lean(0.04); rifle_arms(); return 0.0
    if name.startswith('walk'): i = int(name[4]) - 1; lean(0.06); rifle_arms(); legs(*WALK[i]); return 0.0
    if name.startswith('run'): i = int(name[3]) - 1; lean(0.28); rifle_arms(); legs(*RUN[i]); return 0.0
    if name == 'crouch': lean(0.3); rifle_arms(); crouch_legs(0); return 0.0
    if name.startswith('sneak'): i = int(name[5]) - 1; lean(0.35); rifle_arms(); crouch_legs([1, 0, -1, 0][i]); return 0.0
    if name == 'prone': prone_arms(); prone_legs(0); return -1.5708
    if name.startswith('crawl'): i = int(name[5]) - 1; ph = [1, 0, -1, 0][i]; prone_arms(reach=ph); prone_legs(ph); return -1.5708
    if name == 'shoot': lean(-0.06); recoil_arms(); return 0.0
    if name == 'shootC': lean(0.2); recoil_arms(); crouch_legs(0); return 0.0
    if name == 'shootP': prone_arms(recoil=True); prone_legs(0); return -1.5708
    if name == 'dA1': lean(0.45); dangle_arms(); legs(0.55, 1.1, 0.5, 1.0); aim('Head', (0, -0.6, 0.8)); return 0.0
    if name == 'dA2': lean(0.7); dangle_arms(); kneel_legs(); aim('Head', (0, -0.7, 0.7)); return 0.0
    if name == 'dA3': lean(0.6); dangle_arms(); kneel_legs(); return -0.75
    if name == 'dA4':
        lean(0.1)
        aim('UpperLeg.L', (0.2, 0, -0.98)); aim('LowerLeg.L', (0.1, 0.5, -0.86)); aim('UpperLeg.R', (-0.1, 0.02, -0.99))
        aim('UpperArm.R', (-0.8, -0.2, 0.55)); aim('LowerArm.R', (-0.3, -0.4, 0.86)); aim('UpperArm.L', (0.6, -0.3, -0.74)); aim('LowerArm.L', (0.2, -0.4, -0.9))
        aim('Head', (0.5, -0.3, 0.8)); return -1.5708
    if name == 'dB1': lean(-0.35); fling_arms(); legs(0.15, 0.3, 0.1, 0.35); return 0.3
    if name == 'dB2': lean(-0.2); fling_arms(); legs(0.6, 1.0, 0.45, 0.9); return 0.9
    if name == 'dB3':
        aim('UpperArm.R', (-1, 0.1, 0.1)); aim('LowerArm.R', (-0.9, -0.3, 0.2)); aim('UpperArm.L', (1, 0.1, 0.15)); aim('LowerArm.L', (0.9, -0.2, 0.3))
        aim('UpperLeg.L', (0.1, -0.5, -0.86)); aim('LowerLeg.L', (0, 0.6, -0.8)); return 1.5708
    raise Exception(name)
HUMAN_POSES = ['idle', 'walk1', 'walk2', 'walk3', 'walk4', 'run1', 'run2', 'run3', 'run4', 'crouch', 'sneak1', 'sneak2', 'sneak3', 'sneak4',
               'prone', 'crawl1', 'crawl2', 'crawl3', 'crawl4', 'shoot', 'shootC', 'shootP', 'dA1', 'dA2', 'dA3', 'dA4', 'dB1', 'dB2', 'dB3']
DOG_POSES = ['idle', 'walk1', 'walk2', 'walk3', 'walk4', 'run1', 'run2', 'run3', 'run4', 'bite1', 'bite2', 'd1', 'd2', 'd3']
def dog_pose(name):
    acts = {x.name.split('|')[-1]: x for x in bpy.data.actions}
    if not arm.animation_data: arm.animation_data_create()
    def use(an, t):
        act = acts[an]; arm.animation_data.action = act
        try:
            if hasattr(act, 'slots') and len(act.slots): arm.animation_data.action_slot = act.slots[0]
        except Exception: pass
        f0, f1 = act.frame_range; scn.frame_set(int(round(f0 + (f1 - f0) * t))); vl.update()
    if name == 'idle': use('Idle', 0)
    elif name.startswith('walk'): use('Walk', (int(name[4]) - 1) / 4)
    elif name.startswith('run'): use('Gallop', (int(name[3]) - 1) / 4)
    elif name == 'bite1': use('Attack', .3)
    elif name == 'bite2': use('Attack', .5)
    else: use('Death', {'d1': .3, 'd2': .6, 'd3': 1.0}[name])
    return 0.0
POSES = HUMAN_POSES if KIND == 'human' else DOG_POSES
def set_pose(pi):
    return human_pose(POSES[pi]) if KIND == 'human' else dog_pose(POSES[pi])

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

meta = {'parts': PARTS, 'poses': {}, 'poseNames': POSES}
poses = [POSES.index(x) for x in (sys.argv[-1].split(',') if MODE == 'preview' else POSES)] if MODE == 'preview' else list(range(len(POSES)))
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
