# Player character: Quaternius "Adventurer" (CC0) rendered per body part with clothing and weapon overlays,
# three arm sets (two-handed gun, one-handed in right hand, one-handed in left hand), 2x supersampled.
import bpy, sys, math, json, os, time, bmesh
from mathutils import Vector, Matrix

a = sys.argv[sys.argv.index('--') + 1:]
SRC, OUT, MODE, WDIR = a[0], a[1], a[2], a[3]
SETS = a[4].split(',') if len(a) > 4 else ['rifle', 'oneR', 'oneL']
PREV = a[5].split(',') if len(a) > 5 else None
U = 10.0; KZ = 2.449489743 / 2; SS = 2; P = 2 * math.sqrt(2) * U * SS
RES = 128 * SS; AX, AY = 64 * SS, 100 * SS
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o.name.startswith('Icosphere'): bpy.data.objects.remove(o)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
scn = bpy.context.scene; vl = bpy.context.view_layer

# ---------- materials with a little painted grain so the small sprite reads richer ----------
def mk_mat(name, rgb, rough=0.75, metal=0.0, grain=0.22, gscale=60.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = metal
    add_grain(m, rgb, grain, gscale); return m
def add_grain(m, rgb, amt, scale):
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = scale; nz.inputs['Detail'].default_value = 6
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.inputs['Factor'].default_value = amt
    mix.inputs[6].default_value = (*rgb, 1)
    nt.links.new(nz.outputs['Color'], mix.inputs[7]); nt.links.new(mix.outputs[2], b.inputs['Base Color'])
for m in list(bpy.data.materials):
    if not m.use_nodes: continue
    b = m.node_tree.nodes.get('Principled BSDF')
    if not b: continue
    c = tuple(b.inputs['Base Color'].default_value)[:3]
    b.inputs['Roughness'].default_value = max(0.65, b.inputs['Roughness'].default_value)
    if m.name not in ('Eye', 'Eyebrows'): add_grain(m, c, 0.18 if m.name != 'Skin' else 0.06, 45.0)

piv = bpy.data.objects.new('PIV', None); scn.collection.objects.link(piv); arm.parent = piv; vl.update()
# the model stands away from its own origin: put the hips over the pivot so lying poses stay in frame and sprites centre on the feet
def _bbc():
    dg = bpy.context.evaluated_depsgraph_get(); lo = Vector((1e9, 1e9, 1e9)); hi = -lo
    for o in bpy.data.objects:
        if o.type != 'MESH': continue
        ev = o.evaluated_get(dg); me = ev.to_mesh()
        for v in me.vertices:
            w = ev.matrix_world @ v.co; lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
        ev.to_mesh_clear()
    return (lo + hi) / 2
_hc = _bbc()
for _o in bpy.data.objects:
    if _o.parent is None and _o is not piv and _o.type in ('MESH', 'ARMATURE', 'EMPTY'): _o.parent = piv
for _o in piv.children: _o.location.x -= _hc.x; _o.location.y -= _hc.y
vl.update(); print('recentred by', _hc.x, _hc.y, 'now', _bbc())

# ---------- body parts by bone ----------
def own_part(n):
    if n in ('Neck', 'Head') or n.startswith('Head'): return 'head'
    side = 'L' if n.endswith('.L') else ('R' if n.endswith('.R') else None)
    if side and any(k in n for k in ['UpperArm', 'LowerArm', 'Hand', 'Index', 'Middle', 'Ring', 'Pinky', 'Thumb']): return 'arm' + side
    if side and any(k in n for k in ['UpperLeg', 'LowerLeg', 'Foot', 'Toe']): return 'leg' + side
    return None
def part_of(bname):
    b = arm.data.bones.get(bname)
    while b:
        p = own_part(b.name)
        if p: return p
        b = b.parent
    return 'torso'
HANDB = ('Hand', 'Index', 'Middle', 'Ring', 'Pinky', 'Thumb')
def face_info(o):
    gnames = {g.index: g.name for g in o.vertex_groups}; vb = []
    for v in o.data.vertices:
        best, bw = None, -1
        for g in v.groups:
            if g.weight > bw and g.group in gnames: best, bw = gnames[g.group], g.weight
        vb.append(best or 'Hips')
    out = []
    for f in o.data.polygons:
        cnt = {}
        for vi in f.vertices: cnt[vb[vi]] = cnt.get(vb[vi], 0) + 1
        out.append(max(cnt, key=cnt.get))
    return out
def copy_faces(o, keep, name, offset, mat):
    """copy of mesh o keeping faces where keep(bone) is True, pushed out along normals (stays skinned)"""
    fb = face_info(o); no = o.copy(); no.data = o.data.copy(); no.name = name
    for c in o.users_collection: c.objects.link(no)
    bm = bmesh.new(); bm.from_mesh(no.data); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i, bn in enumerate(fb) if not keep(bn)], context='FACES')
    bm.normal_update()
    k = offset / o.matrix_world.to_scale()[0]
    for v in bm.verts: v.co += v.normal * k
    bm.to_mesh(no.data); bm.free()
    no.data.materials.clear(); no.data.materials.append(mat)
    for p in no.data.polygons: p.material_index = 0
    return no
body = next(o for o in meshes if o.name.endswith('_Body'))
TORSOB = lambda bn: part_of(bn) == 'torso'
UPPERARM = lambda side: (lambda bn: part_of(bn) == 'arm' + side and not bn.startswith(HANDB))
HAND = lambda side: (lambda bn: part_of(bn) == 'arm' + side and bn.startswith(HANDB))
parka = mk_mat('Parka', (0.07, 0.065, 0.045), 0.85, grain=0.12, gscale=20)
rainm = mk_mat('Raincoat', (0.85, 0.62, 0.05), 0.35, grain=0.06, gscale=20)
glovm = mk_mat('Gloves', (0.16, 0.08, 0.035), 0.7, grain=0.1, gscale=20)
hatm = mk_mat('Beanie', (0.12, 0.12, 0.13), 0.95, grain=0.2, gscale=30)
OVER = {}
OVER['jacketT'] = ('torso', [copy_faces(body, TORSOB, 'jacketT', 0.028, parka)])
OVER['rainT'] = ('torso', [copy_faces(body, TORSOB, 'rainT', 0.05, rainm)])
for s in 'LR':
    OVER['jacket' + s] = ('arm' + s, [copy_faces(body, UPPERARM(s), 'jacket' + s, 0.024, parka)])
    OVER['rain' + s] = ('arm' + s, [copy_faces(body, UPPERARM(s), 'rain' + s, 0.042, rainm)])
    OVER['glove' + s] = ('arm' + s, [copy_faces(body, HAND(s), 'glove' + s, 0.012, glovm)])
# beanie: half sphere on the head bone
vl.update()
headm = next(o for o in meshes if o.name.endswith('_Head'))
hp = [headm.matrix_world @ v.co for v in headm.data.vertices]
hz1 = max(p.z for p in hp); hcx = sum(p.x for p in hp) / len(hp); hcy = sum(p.y for p in hp) / len(hp)
hw = (max(p.x for p in hp) - min(p.x for p in hp)) * 0.5
bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=1)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.15], context='VERTS')
for v in bm.verts: v.co = Vector((hcx + v.co.x * hw * 1.02, hcy + v.co.y * hw * 1.08, hz1 - hw * 0.72 + v.co.z * hw * 0.8))
me = bpy.data.meshes.new('hat'); bm.to_mesh(me); bm.free(); me.materials.append(hatm)
hat = bpy.data.objects.new('hat', me); scn.collection.objects.link(hat)
vg = hat.vertex_groups.new(name='Head'); vg.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
md = hat.modifiers.new('Arm', 'ARMATURE'); md.object = arm
hat.parent = arm; hat.matrix_parent_inverse = arm.matrix_world.inverted(); vl.update()
OVER['hat'] = ('head', [hat])

# split the body into parts (same as before)
part_objs = {p: [] for p in ['legL', 'legR', 'torso', 'armL', 'armR', 'head']}
for o in meshes:
    fb = face_info(o); fpart = [part_of(bn) for bn in fb]
    for p in set(fpart):
        no = o.copy(); no.data = o.data.copy(); no.name = p + '__' + o.name
        for c in o.users_collection: c.objects.link(no)
        bm = bmesh.new(); bm.from_mesh(no.data); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i, fp in enumerate(fpart) if fp != p], context='FACES')
        bm.to_mesh(no.data); bm.free(); part_objs[p].append(no)
    bpy.data.objects.remove(o)

# ---------- weapons ----------
def load_glb(path, name, scale, axis, tint=None):
    before = set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    for o in new:
        mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
    for o in [o for o in bpy.data.objects if o not in before and o.type != 'MESH']: bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in new: o.select_set(True)
    vl.objects.active = new[0]
    if len(new) > 1: bpy.ops.object.join()
    o = vl.objects.active; o.name = name
    o.matrix_world = Matrix.Identity(4) @ o.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # turn so that the weapon's long axis is +Y (grip at origin), then scale
    R = {'+x': Matrix.Rotation(-math.pi / 2, 4, 'Z'), '+z': Matrix.Rotation(-math.pi / 2, 4, 'X')}[axis]
    o.data.transform(Matrix.Diagonal((scale, scale, scale, 1)) @ R)
    for m in o.data.materials:
        if m and m.use_nodes:
            b = m.node_tree.nodes.get('Principled BSDF')
            if b: b.inputs['Roughness'].default_value = 0.55
    return o
def prim(name, kind, rgb, dims, metal=0.0):
    bm = bmesh.new()
    if kind == 'rod': bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=dims[0], radius2=dims[1], depth=dims[2]); bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, dims[2] * 0.35))
    elif kind == 'ball': bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=dims[0])
    else: r = bmesh.ops.create_cube(bm, size=1); [setattr(v, 'co', Vector((v.co.x * dims[0], v.co.y * dims[1], v.co.z * dims[2]))) for v in r['verts']]
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); me.materials.append(mk_mat(name + 'M', rgb, 0.5, metal))
    o = bpy.data.objects.new(name, me); scn.collection.objects.link(o)
    if kind == 'rod': o.data.transform(Matrix.Rotation(-math.pi / 2, 4, 'X'))
    return o
S = WDIR
WMODELS = {
    'rifle': lambda: load_glb(S + '/Shotgun_1.glb', 'w_rifle', 0.2, '+x'),
    'shotgun': lambda: load_glb(S + '/Shotgun_ShortStock.glb', 'w_shotgun', 0.19, '+x'),
    'knife': lambda: load_glb(S + '/Knife.glb', 'w_knife', 0.3, '+z'),
    'blade': lambda: load_glb(S + '/Axe_Small.glb', 'w_blade', 0.27, '+z'),
    'blunt': lambda: prim('w_blunt', 'rod', (0.22, 0.2, 0.19), (0.03, 0.04, 0.85), 0.5),
    'gren': lambda: prim('w_gren', 'ball', (0.22, 0.27, 0.14), (0.07,)),
    'molo': lambda: load_glb(S + '/WaterBottle_1.glb', 'w_molo', 0.19, '+z'),
    'charge': lambda: prim('w_charge', 'box', (0.62, 0.12, 0.08), (0.15, 0.08, 0.08)),
}
# molotov: brown glass and a rag
WOBJ = {}
def weapon(kind, side):
    key = kind + side
    if key in WOBJ: return WOBJ[key]
    o = WMODELS[kind](); o.name = 'w_' + key
    if kind == 'molo':
        for m in o.data.materials:
            b = m.node_tree.nodes.get('Principled BSDF'); b.inputs['Base Color'].default_value = (0.3, 0.17, 0.05, 1); b.inputs['Roughness'].default_value = 0.2
    WOBJ[key] = o; return o
def attach(o, side, kind):
    """place weapon o in the hand of the current pose; long axis +Y of the weapon along the chosen direction"""
    vl.update(); mw = arm.matrix_world; hb = arm.pose.bones['Hand.' + side]
    h = mw @ hb.head; t = mw @ hb.tail; hd = (t - h).normalized()
    if kind in ('rifle', 'shotgun'):
        fwd = Vector((0, -1, 0)); pos = h + Vector((0.0, 0.02, -0.02))
        o.matrix_world = Matrix.Translation(pos) @ Matrix.Rotation(math.pi, 4, 'Z')    # weapon +Y -> world -Y
    else:
        d = (hd * 0.35 + Vector((0, -0.25, 1.0))).normalized() if kind in ('knife', 'blade', 'blunt') else hd
        q = Vector((0, 1, 0)).rotation_difference(d)
        pos = h + hd * 0.07
        o.matrix_world = Matrix.Translation(pos) @ q.to_matrix().to_4x4()
    W = o.matrix_world.copy(); o.parent = arm; o.parent_type = 'BONE'; o.parent_bone = 'Hand.' + side; vl.update(); o.matrix_world = W; vl.update()

# ---------- posing ----------
def reset_pose():
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
    vl.update()
def aim(name, d):
    pb = arm.pose.bones.get(name)
    if not pb: return
    vl.update(); mw = arm.matrix_world
    head = mw @ pb.head; tail = mw @ pb.tail
    q = (tail - head).normalized().rotation_difference(Vector(d).normalized())
    R = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head)
    pb.matrix = mw.inverted() @ (R @ (mw @ pb.matrix)); vl.update()
def rifle_arms():
    aim('UpperArm.R', (-0.18, -0.30, -0.94)); aim('LowerArm.R', (0.18, -0.97, -0.12)); aim('Hand.R', (0.1, -1, -0.05))
    aim('UpperArm.L', (0.22, -0.42, -0.88)); aim('LowerArm.L', (-0.55, -0.82, 0.12)); aim('Hand.L', (-0.4, -0.9, 0.05))
def recoil_arms():
    aim('UpperArm.R', (-0.2, -0.12, -0.97)); aim('LowerArm.R', (0.18, -0.88, 0.38)); aim('Hand.R', (0.08, -0.92, 0.38))
    aim('UpperArm.L', (0.22, -0.3, -0.92)); aim('LowerArm.L', (-0.5, -0.78, 0.35)); aim('Hand.L', (-0.35, -0.85, 0.35))
SX = {'R': -1, 'L': 1}
def one_arms(hs, swing=0.0):
    fs = 'L' if hs == 'R' else 'R'; sh, sf = SX[hs], SX[fs]
    aim('UpperArm.' + hs, (sh * 0.2, -0.22 - 0.15 * swing, -0.95)); aim('LowerArm.' + hs, (sh * 0.06, -0.8, -0.55)); aim('Hand.' + hs, (sh * 0.05, -0.9, -0.35))
    aim('UpperArm.' + fs, (sf * 0.2, 0.32 * swing, -0.96)); aim('LowerArm.' + fs, (sf * 0.08, -0.25 + 0.3 * swing, -0.95)); aim('Hand.' + fs, (sf * 0.05, -0.2, -0.97))
def strike_arms(hs):
    fs = 'L' if hs == 'R' else 'R'; sh, sf = SX[hs], SX[fs]
    aim('UpperArm.' + hs, (sh * 0.3, -0.55, 0.78)); aim('LowerArm.' + hs, (sh * 0.1, -0.92, 0.38)); aim('Hand.' + hs, (0, -1, 0.15))
    aim('UpperArm.' + fs, (sf * 0.3, -0.35, -0.88)); aim('LowerArm.' + fs, (sf * 0.1, -0.92, 0.25))
def legs(aL, kL, aR, kR):
    for side, a_, k in (('L', aL, kL), ('R', aR, kR)):
        aim('UpperLeg.' + side, (0, -math.sin(a_), -math.cos(a_))); aim('LowerLeg.' + side, (0, -math.sin(a_ - k), -math.cos(a_ - k)))
        aim('Foot.' + side, (0, -math.cos(a_ - k * .3), -0.25))
WALK = [(0.40, 0.05, -0.35, 0.35), (0.0, 0.05, 0.15, 0.75), (-0.35, 0.35, 0.40, 0.05), (0.15, 0.75, 0.0, 0.05)]
RUN = [(0.75, 0.25, -0.55, 0.8), (0.1, 1.2, 0.25, 0.15), (-0.55, 0.8, 0.75, 0.25), (0.25, 0.15, 0.1, 1.2)]
def lean(a_):
    aim('Torso', (0, -math.sin(a_ * .55), math.cos(a_ * .55))); aim('Chest', (0, -math.sin(a_), math.cos(a_))); aim('Head', (0, -math.sin(a_ * .3) - .05, math.cos(a_ * .3)))
def crouch_legs(ph=0.0):
    for side, sg in (('L', 1), ('R', -1)):
        a_ = 1.08 + 0.28 * ph * sg; aim('UpperLeg.' + side, (sg * 0.13, -math.sin(a_), -math.cos(a_)))
        b_ = a_ - 2.25; aim('LowerLeg.' + side, (0, -math.sin(b_), -math.cos(b_))); aim('Foot.' + side, (0, -1, -0.12))
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
            aim('UpperLeg.' + side, (sg * 0.62, -0.05, -0.78)); aim('LowerLeg.' + side, (sg * 0.12, -0.02, -0.99)); aim('Foot.' + side, (sg * 0.85, -0.35, -0.4))
        else:
            aim('UpperLeg.' + side, (sg * 0.15, 0.0, -0.99)); aim('LowerLeg.' + side, (sg * 0.06, -0.02, -0.998)); aim('Foot.' + side, (sg * 0.45, -0.75, -0.48))
def dangle_arms():
    aim('UpperArm.R', (-0.25, -0.2, -0.95)); aim('LowerArm.R', (-0.05, -0.6, -0.8)); aim('UpperArm.L', (0.25, -0.1, -0.96)); aim('LowerArm.L', (0.1, -0.3, -0.95))
def fling_arms():
    aim('UpperArm.R', (-0.75, 0.35, 0.55)); aim('LowerArm.R', (-0.4, 0.3, 0.86)); aim('UpperArm.L', (0.75, 0.3, 0.6)); aim('LowerArm.L', (0.45, 0.2, 0.87))
def arms_for(st, kind, swing=0.0):
    if st == 'rifle': (recoil_arms if kind == 'shoot' else rifle_arms)()
    else:
        hs = st[3]
        if kind == 'shoot': strike_arms(hs)
        else: one_arms(hs, swing)
def human_pose(name, st):
    reset_pose()
    if name == 'idle': lean(0.04); arms_for(st, 'hold'); return 0.0
    if name.startswith('walk'): i = int(name[4]) - 1; lean(0.06); arms_for(st, 'hold', [0.6, 0, -0.6, 0][i]); legs(*WALK[i]); return 0.0
    if name.startswith('run'): i = int(name[3]) - 1; lean(0.28); arms_for(st, 'hold', [1, 0, -1, 0][i]); legs(*RUN[i]); return 0.0
    if name == 'crouch': lean(0.3); arms_for(st, 'hold'); crouch_legs(0); return 0.0
    if name.startswith('sneak'): i = int(name[5]) - 1; lean(0.35); arms_for(st, 'hold', [0.3, 0, -0.3, 0][i]); crouch_legs([1, 0, -1, 0][i]); return 0.0
    if name == 'prone': prone_arms(); prone_legs(0); return -1.5708
    if name.startswith('crawl'): i = int(name[5]) - 1; ph = [1, 0, -1, 0][i]; prone_arms(reach=ph); prone_legs(ph); return -1.5708
    if name == 'shoot': lean(-0.06); arms_for(st, 'shoot'); return 0.0
    if name == 'shootC': lean(0.2); arms_for(st, 'shoot'); crouch_legs(0); return 0.0
    if name == 'shootP': prone_arms(recoil=True); prone_legs(0); return -1.5708
    if name == 'dA1': lean(0.45); dangle_arms(); legs(0.55, 1.1, 0.5, 1.0); aim('Head', (0, -0.6, 0.8)); return 0.0
    if name == 'dA2': lean(0.7); dangle_arms(); kneel_legs(); aim('Head', (0, -0.7, 0.7)); return 0.0
    if name == 'dA3': lean(0.6); dangle_arms(); kneel_legs(); return -0.75
    if name == 'dA4':
        lean(0.1); aim('UpperLeg.L', (0.2, 0, -0.98)); aim('LowerLeg.L', (0.1, 0.5, -0.86)); aim('UpperLeg.R', (-0.1, 0.02, -0.99))
        aim('UpperArm.R', (-0.8, -0.2, 0.55)); aim('LowerArm.R', (-0.3, -0.4, 0.86)); aim('UpperArm.L', (0.6, -0.3, -0.74)); aim('LowerArm.L', (0.2, -0.4, -0.9))
        aim('Head', (0.5, -0.3, 0.8)); return -1.5708
    if name == 'dB1': lean(-0.35); fling_arms(); legs(0.15, 0.3, 0.1, 0.35); return 0.3
    if name == 'dB2': lean(-0.2); fling_arms(); legs(0.6, 1.0, 0.45, 0.9); return 0.9
    if name == 'dB3':
        aim('UpperArm.R', (-1, 0.1, 0.1)); aim('LowerArm.R', (-0.9, -0.3, 0.2)); aim('UpperArm.L', (1, 0.1, 0.15)); aim('LowerArm.L', (0.9, -0.2, 0.3))
        aim('UpperLeg.L', (0.1, -0.5, -0.86)); aim('LowerLeg.L', (0, 0.6, -0.8)); return 1.5708
    raise Exception(name)
POSES = ['idle', 'walk1', 'walk2', 'walk3', 'walk4', 'run1', 'run2', 'run3', 'run4', 'crouch', 'sneak1', 'sneak2', 'sneak3', 'sneak4',
         'prone', 'crawl1', 'crawl2', 'crawl3', 'crawl4', 'shoot', 'shootC', 'shootP', 'dA1', 'dA2', 'dA3', 'dA4', 'dB1', 'dB2', 'dB3']
ONE_POSES = POSES[:22]
# parts rendered in each arm set: name -> (base part, objects)
def set_parts(st):
    P_ = {}
    if st == 'rifle':
        for p in ['legL', 'legR', 'torso', 'armL', 'armR', 'head']: P_[p] = (p, part_objs[p])
        for k in ['hat', 'jacketT', 'rainT']: P_[k] = OVER[k]
    else:
        for p in ['armL', 'armR']: P_[p] = (p, part_objs[p])
    for s in 'LR':
        for k in ['jacket', 'rain', 'glove']: P_[k + s] = OVER[k + s]
    if st == 'rifle':
        P_['w_rifle'] = ('armR', [weapon('rifle', 'R')]); P_['w_shotgun'] = ('armR', [weapon('shotgun', 'R')])
    else:
        hs = st[3]
        for k in ['knife', 'blade', 'blunt', 'gren', 'molo', 'charge']: P_['w_' + k] = ('arm' + hs, [weapon(k, hs)])
    return P_

# ---------- scene, camera, light ----------
scn.render.engine = 'CYCLES'; scn.cycles.device = 'CPU'; scn.cycles.use_denoising = False
scn.render.film_transparent = True; scn.render.resolution_x = scn.render.resolution_y = RES; scn.render.resolution_percentage = 100
scn.render.use_persistent_data = True; scn.render.image_settings.file_format = 'PNG'; scn.render.image_settings.color_mode = 'RGBA'; scn.render.image_settings.color_depth = '16'
try: scn.view_settings.view_transform = 'Standard'
except Exception as e: print('VT', e)
world = bpy.data.worlds.new('W'); scn.world = world; world.use_nodes = True
bgn = world.node_tree.nodes['Background']; bgn.inputs['Color'].default_value = (0.62, 0.66, 0.74, 1); bgn.inputs['Strength'].default_value = 1.0
sun = bpy.data.lights.new('Sun', 'SUN'); sun.energy = 4.6; sun.angle = math.radians(4)
so = bpy.data.objects.new('Sun', sun); scn.collection.objects.link(so)
Ld = Vector((0.3, -0.6, 1)).normalized(); so.rotation_mode = 'QUATERNION'; so.rotation_quaternion = (-Ld).to_track_quat('-Z', 'Y')
cam = bpy.data.cameras.new('Cam'); cam.type = 'ORTHO'; cam.ortho_scale = RES / P
co = bpy.data.objects.new('Cam', cam); scn.collection.objects.link(co); scn.camera = co
e = math.radians(30)
Dc = Vector((math.cos(e) * math.cos(math.radians(45)), math.cos(e) * math.sin(math.radians(45)), math.sin(e)))
Up = Vector((-math.sin(e) * math.cos(math.radians(45)), -math.sin(e) * math.sin(math.radians(45)), math.cos(e)))
co.location = Up * ((AY - RES / 2) / P) + Dc * 30
co.rotation_mode = 'QUATERNION'; co.rotation_quaternion = (-Dc).to_track_quat('-Z', 'Y'); cam.clip_end = 100
pm = bpy.data.materials.new('POS'); pm.use_nodes = True; nt = pm.node_tree; nt.nodes.clear()
geo = nt.nodes.new('ShaderNodeNewGeometry'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sep.inputs[0])
def math_(op, a_, b_):
    n = nt.nodes.new('ShaderNodeMath'); n.operation = op
    for i, v in enumerate((a_, b_)):
        if isinstance(v, (int, float)): n.inputs[i].default_value = v
        else: nt.links.new(v, n.inputs[i])
    return n.outputs[0]
xy = math_('ADD', sep.outputs[0], sep.outputs[1])
dep = math_('ADD', math_('MULTIPLY', xy, U), math_('MULTIPLY', sep.outputs[2], U * KZ))
Rr = math_('MULTIPLY', math_('ADD', dep, 64), 1 / 128); Gc = math_('MULTIPLY', sep.outputs[2], U * KZ / 64)
cc = nt.nodes.new('ShaderNodeCombineColor'); nt.links.new(Rr, cc.inputs[0]); nt.links.new(Gc, cc.inputs[1]); cc.inputs[2].default_value = 0.5
em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(cc.outputs[0], em.inputs[0]); em.inputs[1].default_value = 1
out = nt.nodes.new('ShaderNodeOutputMaterial'); nt.links.new(em.outputs[0], out.inputs[0])
def color_mode(): vl.material_override = None; scn.cycles.samples = 24; scn.cycles.filter_width = 1.5; scn.cycles.max_bounces = 3
def pos_mode(): vl.material_override = pm; scn.cycles.samples = 1; scn.cycles.filter_width = 0.01; scn.cycles.max_bounces = 0
ALLOBJ = lambda: [o for o in bpy.data.objects if o.type == 'MESH']
def min_z(objs):
    dg = bpy.context.evaluated_depsgraph_get(); mz = 1e9
    for o in objs:
        ev = o.evaluated_get(dg); me = ev.to_mesh(); mw = ev.matrix_world
        for v in me.vertices: mz = min(mz, (mw @ v.co).z)
        ev.to_mesh_clear()
    return mz
def hipc():
    # keep the hips over the sprite anchor whatever the pose does to the skeleton
    h = (arm.matrix_world @ arm.pose.bones['UpperLeg.L'].head + arm.matrix_world @ arm.pose.bones['UpperLeg.R'].head) / 2
    piv.location.x -= h.x; piv.location.y -= h.y; vl.update()
    h = (arm.matrix_world @ arm.pose.bones["UpperLeg.L"].head + arm.matrix_world @ arm.pose.bones["UpperLeg.R"].head) / 2
    piv.location.x -= h.x; piv.location.y -= h.y; vl.update()
def E(w): return [round(w.y * U, 2), round(w.x * U, 2), round(w.z * U * KZ, 2)]
def joints(st):
    vl.update(); mw = arm.matrix_world; J = {}
    for k, bn in {'head': 'Neck', 'armL': 'UpperArm.L', 'armR': 'UpperArm.R', 'legL': 'UpperLeg.L', 'legR': 'UpperLeg.R', 'torsoC': 'Chest', 'handL': 'Hand.L', 'handR': 'Hand.R'}.items():
        J[k] = E(mw @ arm.pose.bones[bn].head)
    if st == 'rifle':
        J['muzzle'] = E(weapon('rifle', 'R').matrix_world @ Vector((0, 0.78, 0.03)))
        J['muzzleS'] = E(weapon('shotgun', 'R').matrix_world @ Vector((0, 0.78, 0.03)))
    return J

meta = {'parts': ['legL', 'legR', 'torso', 'armL', 'armR', 'head'], 'poseNames': POSES, 'sets': {}}
t0 = time.time()
for st in SETS:
    # weapon attachment in the set's reference pose
    piv.matrix_world = Matrix.Identity(4); vl.update(); reset_pose(); arms_for(st, 'hold')
    sp = set_parts(st)
    for name, (base, objs) in sp.items():
        if name.startswith('w_'): attach(objs[0], 'R' if base == 'armR' else 'L', name[2:])
    poses = ONE_POSES if st != 'rifle' else POSES
    if PREV: poses = [p for p in PREV if p in poses]
    meta['sets'][st] = {'parts': {k: v[0] for k, v in sp.items()}, 'poses': [POSES.index(p) for p in poses], 'joints': {}}
    allobjs = ALLOBJ()
    for pn in poses:
        pose = POSES.index(pn)
        piv.matrix_world = Matrix.Identity(4); vl.update()
        fall = human_pose(pn, st)
        piv.matrix_world = Matrix.Rotation(math.pi / 2, 4, 'Z') @ Matrix.Rotation(-fall, 4, 'X'); vl.update()
        lift = -min_z([o for p in part_objs.values() for o in p]); piv.location = (0, 0, lift); vl.update()
        hipc();         meta['sets'][st]['joints'][pose] = joints(st)
        if os.environ.get('DBG'): print('DBG', [(o.name, round(min_z([o]),2)) for p in part_objs.values() for o in p], pn, 'bbc', _bbc(), 'hips', arm.matrix_world @ arm.pose.bones['UpperLeg.L'].head, 'pivloc', piv.location, 'arm', arm.location, arm.matrix_world.translation)
        for d in ([1, 7] if PREV else range(8)):
            piv.matrix_world = Matrix.Translation((0, 0, lift)) @ Matrix.Rotation(math.pi / 2 - d * math.pi / 4, 4, 'Z') @ Matrix.Rotation(-fall, 4, 'X'); vl.update(); hipc()
            if PREV:
                color_mode()
                show = set(o.name for k, (b, objs) in sp.items() if k in ('armL', 'armR', 'hat', 'jacketT', 'jacketL', 'jacketR', 'gloveL', 'gloveR') or k in ('w_rifle', 'w_knife', 'w_gren') for o in objs) | set(o.name for p in ('legL', 'legR', 'torso', 'head') for o in part_objs[p]) | {'hat', 'jacketT'}
                for o in allobjs: o.hide_render = o.name not in show
                scn.render.filepath = f'{OUT}/prev_{st}_{pn}_{d}.png'; bpy.ops.render.render(write_still=True); continue
            for mode in ('c', 'p'):
                color_mode() if mode == 'c' else pos_mode()
                for name, (base, objs) in sp.items():
                    keep = set(id(o) for o in objs)
                    for o in allobjs: o.hide_render = id(o) not in keep
                    scn.render.filepath = f'{OUT}/{st}_{mode}_{pose}_{d}_{name}.png'
                    bpy.ops.render.render(write_still=True)
        print('set', st, 'pose', pn, round(time.time() - t0, 1), flush=True)
json.dump(meta, open(f'{OUT}/meta_{"_".join(SETS)}.json', 'w'))
print('ALLDONE', flush=True)
