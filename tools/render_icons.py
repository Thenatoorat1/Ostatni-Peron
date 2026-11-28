# Renders inventory icons for the grenade and the demolition charge (transparent PNG).
import bpy, math, sys
from mathutils import Vector
OUT = sys.argv[sys.argv.index('--') + 1]

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 96; sc.cycles.use_denoising = True
    sc.render.film_transparent = True; sc.render.resolution_x = 512; sc.render.resolution_y = 512
    sc.view_settings.view_transform = 'Standard'
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.55, 0.55, 0.6, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.25
    return sc

def mat(name, col, rough=0.6, metal=0.0, bump=None, wear=None):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; N = nt.nodes; L = nt.links
    b = N['Principled BSDF']; b.inputs['Base Color'].default_value = col + (1,); b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = metal
    tc = N.new('ShaderNodeTexCoord')
    if wear:  # worn paint: noise mixes in bare metal
        nz = N.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = wear[0]; nz.inputs['Detail'].default_value = 8
        L.new(tc.outputs['Object'], nz.inputs['Vector'])
        cr = N.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].position = wear[1]; cr.color_ramp.elements[1].position = wear[1] + 0.04
        cr.color_ramp.elements[0].color = col + (1,); cr.color_ramp.elements[1].color = (0.33, 0.31, 0.28, 1)
        L.new(nz.outputs['Fac'], cr.inputs['Fac']); L.new(cr.outputs['Color'], b.inputs['Base Color'])
    if bump:
        bn = N.new('ShaderNodeBump'); bn.inputs['Strength'].default_value = bump[1]; bn.inputs['Distance'].default_value = 0.06
        L.new(bump[0](N, L, tc), bn.inputs['Height']); L.new(bn.outputs['Normal'], b.inputs['Normal'])
    return m

def grooves(N, L, tc):
    # segmented "pineapple" body: grooves around (z) and along (angle)
    sep = N.new('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Object'], sep.inputs[0])
    at = N.new('ShaderNodeMath'); at.operation = 'ARCTAN2'; L.new(sep.outputs['Y'], at.inputs[0]); L.new(sep.outputs['X'], at.inputs[1])
    a8 = N.new('ShaderNodeMath'); a8.operation = 'MULTIPLY'; a8.inputs[1].default_value = 8 / (2 * math.pi) * 2 * math.pi; L.new(at.outputs[0], a8.inputs[0])
    s1 = N.new('ShaderNodeMath'); s1.operation = 'COSINE'; L.new(a8.outputs[0], s1.inputs[0])
    zz = N.new('ShaderNodeMath'); zz.operation = 'MULTIPLY'; zz.inputs[1].default_value = 5.2 * math.pi; L.new(sep.outputs['Z'], zz.inputs[0])
    s2 = N.new('ShaderNodeMath'); s2.operation = 'COSINE'; L.new(zz.outputs[0], s2.inputs[0])
    p1 = N.new('ShaderNodeMath'); p1.operation = 'POWER'; p1.inputs[1].default_value = 0.25
    ab1 = N.new('ShaderNodeMath'); ab1.operation = 'ABSOLUTE'; L.new(s1.outputs[0], ab1.inputs[0]); L.new(ab1.outputs[0], p1.inputs[0])
    p2 = N.new('ShaderNodeMath'); p2.operation = 'POWER'; p2.inputs[1].default_value = 0.25
    ab2 = N.new('ShaderNodeMath'); ab2.operation = 'ABSOLUTE'; L.new(s2.outputs[0], ab2.inputs[0]); L.new(ab2.outputs[0], p2.inputs[0])
    mn = N.new('ShaderNodeMath'); mn.operation = 'MINIMUM'; L.new(p1.outputs[0], mn.inputs[0]); L.new(p2.outputs[0], mn.inputs[1])
    return mn.outputs[0]

def lights_cam(target, dist, ortho):
    sc = bpy.context.scene
    for (loc, e, s) in [((3, -4, 6), 260, 3), ((-5, -2, 3), 60, 4), ((0, 5, 4), 140, 3)]:
        l = bpy.data.lights.new('l', 'AREA'); l.energy = e; l.size = s
        o = bpy.data.objects.new('l', l); o.location = loc; sc.collection.objects.link(o)
        o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    c = bpy.data.cameras.new('c'); c.type = 'ORTHO'; c.ortho_scale = ortho
    co = bpy.data.objects.new('c', c); sc.collection.objects.link(co); sc.camera = co
    el, az = math.radians(28), math.radians(-35)
    co.location = Vector(target) + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * dist
    co.rotation_euler = (Vector(target) - co.location).to_track_quat('-Z', 'Y').to_euler()

def link(o, m=None):
    bpy.context.collection.objects.link(o) if o.name not in bpy.context.collection.objects else None
    if m: o.data.materials.append(m)
    return o

def grenade():
    reset()
    olive = mat('olive', (0.20, 0.24, 0.12), 0.55, 0.15, bump=(grooves, 1.0), wear=(9, 0.73))
    steel = mat('steel', (0.55, 0.55, 0.52), 0.35, 1.0)
    brass = mat('brass', (0.62, 0.5, 0.25), 0.3, 1.0)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.5); body = bpy.context.object; body.scale = (1, 1, 1.22)
    bpy.ops.object.shade_smooth(); body.data.materials.append(olive)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.16, depth=0.12, location=(0, 0, 0.63)); bpy.context.object.data.materials.append(olive)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.12, depth=0.2, location=(0, 0, 0.75)); fz = bpy.context.object; fz.data.materials.append(steel)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.05, depth=0.12, location=(0, 0, 0.9)); bpy.context.object.data.materials.append(brass)
    # spoon (lever) running down the side
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.2, 0, 0.8)); sp = bpy.context.object; sp.scale = (0.26, 0.11, 0.025); sp.data.materials.append(steel)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.39, 0, 0.45)); sp2 = bpy.context.object; sp2.scale = (0.03, 0.11, 0.62); sp2.rotation_euler = (0, math.radians(-14), 0); sp2.data.materials.append(steel)
    # pin and pull ring
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.018, depth=0.42, location=(0, -0.05, 0.76), rotation=(math.radians(90), 0, 0)); bpy.context.object.data.materials.append(steel)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.17, minor_radius=0.022, location=(-0.02, -0.38, 0.76), rotation=(0, math.radians(70), math.radians(10))); bpy.context.object.data.materials.append(steel)
    lights_cam((0, 0, 0.15), 6, 1.95)
    bpy.context.scene.render.filepath = OUT + 'ic_grenade.png'; bpy.ops.render.render(write_still=True)

def charge():
    reset()
    bpy.context.scene.render.resolution_x = 1024
    canvas = mat('canvas', (0.26, 0.25, 0.16), 0.85, 0.0, wear=(4, 0.8))
    block = mat('block', (0.72, 0.68, 0.55), 0.7)
    tape = mat('tape', (0.12, 0.12, 0.12), 0.45)
    red = mat('red', (0.55, 0.06, 0.04), 0.5); blue = mat('blue', (0.05, 0.12, 0.45), 0.5)
    alu = mat('alu', (0.7, 0.7, 0.72), 0.25, 1.0)
    # two blocks of explosive in a canvas wrap
    for y in (-0.22, 0.22):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, 0)); b = bpy.context.object; b.scale = (1.8, 0.42, 0.36)
        bv = b.modifiers.new('b', 'BEVEL'); bv.width = 0.04; bv.segments = 3; b.data.materials.append(canvas)
    for x in (-0.55, 0.55):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, 0)); t = bpy.context.object; t.scale = (0.16, 0.9, 0.39); t.data.materials.append(tape)
    # exposed end of a block and a blasting cap pushed into it
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.92, 0.22, 0)); e = bpy.context.object; e.scale = (0.04, 0.38, 0.32); e.data.materials.append(block)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.045, depth=0.34, location=(1.06, 0.22, 0.02), rotation=(0, math.radians(90), 0)); bpy.context.object.data.materials.append(alu)
    # wires
    for mtl, dz in ((red, 0.03), (blue, -0.02)):
        cu = bpy.data.curves.new('w', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 0.018; cu.bevel_resolution = 3
        sp = cu.splines.new('BEZIER'); pts = [(1.22, 0.22, 0.02 + dz), (1.5, 0.1, 0.3 + dz), (1.35, -0.4, 0.5 + dz), (1.75, -0.6, 0.3)]
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts): bp.co = p; bp.handle_left_type = bp.handle_right_type = 'AUTO'
        o = bpy.data.objects.new('w', cu); bpy.context.collection.objects.link(o); o.data.materials.append(mtl)
    lights_cam((0.15, 0, 0), 8, 4.2)
    bpy.context.scene.render.filepath = OUT + 'ic_charge.png'; bpy.ops.render.render(write_still=True)

grenade(); charge()
