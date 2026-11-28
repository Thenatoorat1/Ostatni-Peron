"""Build the static world for the sprite demo from the Magnific images.
Engine projection: sx=(x-y)*2+OX, sy=x+y-2z+OY ; depth=x+y+z ; units 10/m horizontally."""
import json, math, numpy as np
from PIL import Image
from scipy import ndimage

W = D = 300; H = 180
BW = (W + D) * 2 + 8; BH = (W + D) + 2 * H + 8; OX = 2 * D; OY = 2 * H
MAG = 'mag/'; OUT = 'assets/'
rng = np.random.default_rng(7)

def key_magenta(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = np.minimum(R, B) - G
    alpha = 1 - np.clip((m - 55) / 50, 0, 1)
    alpha[(G > 140)] = 1
    # despill
    sp = np.clip(m - 10, 0, None) * (1 - alpha * 0.6)
    a[..., 0] -= sp * 0.55; a[..., 2] -= sp * 0.55
    a = np.clip(a, 0, 255)
    # remove small specks
    lab, n = ndimage.label(alpha > 0.5)
    if n:
        sizes = ndimage.sum(np.ones_like(alpha), lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sizes > 400
        alpha = alpha * keep[lab]
    return a, alpha

def comps(alpha, k):
    lab, n = ndimage.label(ndimage.binary_dilation(alpha > 0.5, iterations=3))
    sl = ndimage.find_objects(lab)
    boxes = [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in sl]
    boxes = [b for b in boxes if (b[1] - b[0]) * (b[3] - b[2]) > 4000]
    boxes.sort(key=lambda b: ((b[0] + b[1]) // 2 // 500, (b[2] + b[3]) // 2))
    assert len(boxes) >= k, len(boxes)
    return boxes[:k]

def crop_scale(rgb, alpha, box, scale):
    y0, y1, x0, x1 = box
    r = rgb[y0:y1, x0:x1]; al = alpha[y0:y1, x0:x1]
    pre = np.dstack([r * al[..., None], al * 255]).astype(np.float32)
    h, w = al.shape; nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
    chans = [np.asarray(Image.fromarray(pre[..., c]).resize((nw, nh), Image.LANCZOS)) for c in range(4)]
    A = np.clip(chans[3] / 255, 0, 1)
    rgbs = np.dstack([np.clip(chans[c] / np.maximum(A, 1e-3), 0, 255) for c in range(3)])
    mask = A > 0.5
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1)); big = np.argmax(sizes) + 1
        keepl = np.zeros(n + 1, bool); keepl[1:] = sizes >= max(6, sizes.max() * 0.02)
        mask = keepl[lab]
    ys, xs = np.nonzero(mask)
    yy0, yy1, xx0, xx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgbs[yy0:yy1, xx0:xx1], mask[yy0:yy1, xx0:xx1]

def sharpen(rgb, amt=0.35):
    bl = ndimage.gaussian_filter(rgb, sigma=(0.8, 0.8, 0))
    rgb = np.clip(rgb + (rgb - bl) * amt, 0, 255)
    s = np.clip(np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1] - 4, 0, None)
    rgb[..., 0] -= s; rgb[..., 2] -= s
    return rgb

objects = []   # dict entries
def add_object(name, mat, rgb, mask, shape, anchor, ex=False, thick=5, blocks=True, interior=None, sx_anchor=None):
    """shape: ('box', x1, y1, w, d, Htop) front corner + size, or ('cyl', cx, cy, R)"""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    by = ys.max(); bx = int(np.round(xs[ys >= by - 1].mean()))
    if shape[0] == 'box':
        _, X1, Y1, Wd, Dd, Ht = shape
        fsx = (X1 - Y1) * 2 + OX; fsy = X1 + Y1 + OY
        px0 = fsx - bx; py0 = fsy - by
    else:
        _, CX, CY, R = shape
        fsx = (CX - CY) * 2 + OX; fsy = CX + CY + OY + math.sqrt(2) * R
        px0 = fsx - bx; py0 = round(fsy - by)
    px0 = int(round(px0)); py0 = int(round(py0))
    yy, xx = np.mgrid[0:h, 0:w]
    sx = xx + px0 + 0.5; sy = yy + py0 + 0.5
    x0r = ((sx - OX) / 2 + (sy - OY)) / 2; y0r = ((sy - OY) - (sx - OX) / 2) / 2
    if shape[0] == 'box':
        t = np.minimum(X1 - x0r, Y1 - y0r)
        if Ht: t = np.minimum(t, Ht)
    else:
        # vertical cylinder: ray (x0r+t, y0r+t, t); |(x0r-CX+t, y0r-CY+t)| = R  -> largest t
        ax_ = x0r - CX; ay_ = y0r - CY
        A = 2.0; Bq = 2 * (ax_ + ay_); Cq = ax_ ** 2 + ay_ ** 2 - R ** 2
        disc = Bq * Bq - 4 * A * Cq
        tp = (-Bq + np.sqrt(np.maximum(disc, 0))) / (2 * A)
        tc = -(ax_ + ay_) / 2
        t = np.where(disc > 0, tp, tc)
    t = np.maximum(t, 0)
    dep = x0r + y0r + 3 * t; zh = t
    layers = [(rgb, mask, dep, zh)]
    if interior:
        layers.append(interior(x0r, y0r, t, mask))
    objects.append(dict(name=name, mat=mat, ex=ex, x0=px0, y0=py0, w=w, h=h, layers=layers, thick=thick, shape=shape, blocks=blocks))

def build_interior(X1, Y1, Wd, Dd, floors, Htop, wall=3):
    bx0 = X1 - Wd + wall; by0 = Y1 - Dd + wall
    def f(x0r, y0r, tfront, mask):
        h, w = mask.shape
        best = np.full((h, w), -1e9); kind = np.zeros((h, w), np.int8)
        for fz in floors:
            t = np.full((h, w), float(fz)); px = x0r + t; py = y0r + t
            ok = (px >= bx0) & (px <= X1 - wall) & (py >= by0) & (py <= Y1 - wall) & (t < tfront - 1.5)
            upd = ok & (t > best); best[upd] = t[upd]; kind[upd] = 1
        t = bx0 - x0r; py = y0r + t
        ok = (py >= by0) & (py <= Y1 - wall) & (t >= 0) & (t <= Htop) & (t < tfront - 1.5)
        upd = ok & (t > best); best[upd] = t[upd]; kind[upd] = 2
        t = by0 - y0r; px = x0r + t
        ok = (px >= bx0) & (px <= X1 - wall) & (t >= 0) & (t <= Htop) & (t < tfront - 1.5)
        upd = ok & (t > best); best[upd] = t[upd]; kind[upd] = 3
        valid = mask & (kind > 0)
        n = rng.normal(0, 1, (h, w))
        col = np.zeros((h, w, 3))
        z = np.maximum(best, 0)
        # storey-relative height for gloom
        fl = np.array(floors + [Htop])
        rel = np.zeros_like(z)
        for i in range(len(fl) - 1):
            sel = (z >= fl[i]) & (z < fl[i + 1]); rel[sel] = (z[sel] - fl[i]) / max(1, fl[i + 1] - fl[i])
        wallc = np.array([92, 82, 70]); floorc = np.array([70, 60, 50])
        col[kind == 1] = floorc; col[kind == 2] = wallc * 0.82; col[kind == 3] = wallc
        col *= (0.72 + 0.2 * (kind[..., None] != 1) * (1 - rel[..., None]) * 0.5)[...]
        col += n[..., None] * 7
        speck = (rng.random((h, w)) < 0.06) & (kind == 1)
        col[speck] = [120, 108, 92]
        col = np.clip(col * 0.78, 0, 255)
        dep = x0r + y0r + 3 * z
        imt = np.where(kind == 1, 6, 5).astype(np.uint8)
        return (col, valid, dep, z, imt)
    return f

def place_box_img(fname, box, scale, name, mat, X1, Y1, Htop=None, interior=None, blocks=True, ex=False, flatroof=False, floors=None):
    rgb, al = key_magenta(MAG + fname) if isinstance(fname, str) else fname
    if box is None:
        ys, xs = np.nonzero(al > 0.5); box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    r, m = crop_scale(rgb, al, box, scale)
    r = sharpen(r)
    h, w = m.shape; ys, xs = np.nonzero(m)
    by = ys.max(); bx = xs[ys >= by - 1].mean()
    low = ys > h * 0.62
    lx = xs[low].min(); rx = xs[low].max()
    Wd = (bx - lx) / 2; Dd = (rx - bx) / 2
    Ht = None
    if flatroof:
        col_top = ys[np.abs(xs - bx) < 1.5].min()
        Ht = ((by - col_top) - (Wd + Dd)) / 2
    shape = ('box', X1, Y1, Wd, Dd, Ht if flatroof else Htop)
    inter = None
    if interior:
        Hmax = Ht if Ht else (by - ys.min() - (Wd + Dd)) / 2
        inter = build_interior(X1, Y1, Wd, Dd, interior(Hmax), Hmax)
    add_object(name, mat, r, m, shape, None, ex=ex, thick=3 if interior else 5, blocks=blocks, interior=inter)
    print(f'{name}: footprint {Wd:.0f}x{Dd:.0f} units, h~{(by - ys.min() - (Wd + Dd)) / 2:.0f}, img {w}x{h}')
    return Wd, Dd

def place_cyl_img(src, box, scale, name, mat, CX, CY, R, ex=False, blocks=True):
    rgb, al = src
    r, m = crop_scale(rgb, al, box, scale); r = sharpen(r)
    add_object(name, mat, r, m, ('cyl', CX, CY, R), None, ex=ex, blocks=blocks)

# ---------------- layout ----------------
# buildings behind the road (small y), road y 120..176, plaza in front
kam = key_magenta(MAG + 'budynek_kamienica.png')
pav = key_magenta(MAG + 'budynek_pawilon.png')
ys, xs = np.nonzero(kam[1] > .5); kw = xs.max() - xs.min()
place_box_img(kam, None, 380 / kw, 'Kamienica', 2, 128, 104, interior=lambda Hm: [0, round(Hm * 0.42)])
ys, xs = np.nonzero(pav[1] > .5); pw = xs.max() - xs.min()
place_box_img(pav, None, 280 / pw, 'Pawilon handlowy', 2, 262, 98, flatroof=True, interior=lambda Hm: [0])
car = key_magenta(MAG + 'wrak_auta.png')
ys, xs = np.nonzero(car[1] > .5); cw = xs.max() - xs.min()
place_box_img(car, None, 112 / cw, 'Wrak samochodu', 4, 178, 162)

props = key_magenta(MAG + 'rekwizyty.png')
pb = comps(props[1], 6)  # blue barrel, red barrel, sandbags, barrier, crates, lamp
def w_of(b): return b[3] - b[2]
def h_of(b): return b[1] - b[0]
bar_s = 18 / w_of(pb[0])
for (cx, cy, red) in [(150, 214, True), (160, 206, False), (156, 222, False), (98, 236, True), (230, 200, False)]:
    place_cyl_img(props, pb[1] if red else pb[0], bar_s, 'Beczka z paliwem' if red else 'Beczka', 9 if red else 4, cx, cy, 3.2, ex=red)
sb = pb[2]
for (x1, y1) in [(214, 246), (196, 252)]:
    place_box_img(props, sb, 46 / w_of(sb), 'Worki z piaskiem', 10, x1, y1)
br = pb[3]
for (x1, y1) in [(122, 196), (168, 190), (88, 204)]:
    place_box_img(props, br, 54 / w_of(br), 'Bariera betonowa', 2, x1, y1)
cr = pb[4]
place_box_img(props, cr, 34 / w_of(cr), 'Skrzynie', 3, 132, 230)
lp = pb[5]
place_cyl_img(props, lp, 118 / h_of(lp), 'Latarnia', 4, 196, 116, 1.5)

veg = key_magenta(MAG + 'roslinnosc_gruz.png')
vb = comps(veg[1], 6)  # grass x3, bush x2, rubble
for i, (cx, cy) in enumerate([(70, 260), (250, 280), (40, 200), (180, 290), (276, 230), (110, 286), (240, 170)]):
    b = vb[i % 3]; place_cyl_img(veg, b, 30 / w_of(b), 'Trawa', 5, cx, cy, 2, blocks=False)
for (cx, cy, i) in [(56, 228, 3), (280, 196, 4), (24, 286, 3), (150, 116, 4)]:
    b = vb[i]; place_cyl_img(veg, b, 66 / w_of(b), 'Krzak', 5, cx, cy, 5, blocks=False)
rb = vb[5]
place_cyl_img(veg, rb, 70 / w_of(rb), 'Gruz', 2, 140, 114, 7)

pine = key_magenta(MAG + 'sosna.png'); ys, xs = np.nonzero(pine[1] > .5); ph = ys.max() - ys.min()
birch = key_magenta(MAG + 'brzoza.png'); ys2, xs2 = np.nonzero(birch[1] > .5); bh = ys2.max() - ys2.min()
for (cx, cy, hh) in [(292, 40, 300), (286, 120, 270), (20, 130, 280), (14, 40, 300), (173, 42, 260), (62, 6, 250)]:
    place_cyl_img(pine, (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1), hh / ph, 'Sosna', 5, cx, cy, 2.2)
for (cx, cy, hh) in [(40, 250, 200), (282, 264, 180)]:
    place_cyl_img(birch, (ys2.min(), ys2.max() + 1, xs2.min(), xs2.max() + 1), hh / bh, 'Brzoza', 5, cx, cy, 1.6)

# ---------------- enterable buildings: door objects and cutaway storey layers ----------------
# Each building gets one extra layer per storey: the scene ray-cast as boxes (floor slabs, back walls up to the ceiling,
# front walls cut to a low stub like in Fallout, stairs, furniture). The engine shows that layer instead of the facade
# while the player is inside on that storey.
def obj_grid(o):
    h, w = o['h'], o['w']
    yy, xx = np.mgrid[0:h, 0:w]
    sx = xx + o['x0'] + 0.5; sy = yy + o['y0'] + 0.5
    return ((sx - OX) / 2 + (sy - OY)) / 2, ((sy - OY) - (sx - OX) / 2) / 2

def box_cast(x0r, y0r, boxes):
    """boxes: list of (x0,y0,z0,x1,y1,z1, style). Returns t, face (0 +x,1 +y,2 top), box index."""
    h, w = x0r.shape
    best = np.full((h, w), -1.0); face = np.zeros((h, w), np.int8); bi = np.full((h, w), -1, np.int32)
    for i, (bx0, by0, bz0, bx1, by1, bz1, st) in enumerate(boxes):
        tx1 = bx1 - x0r; ty1 = by1 - y0r
        thi = np.minimum(np.minimum(tx1, ty1), bz1)
        tlo = np.maximum(np.maximum(np.maximum(bx0 - x0r, by0 - y0r), bz0), 0)
        hit = (tlo <= thi + 1e-6) & (thi >= 0) & (thi > best)
        f = np.where((tx1 <= ty1) & (tx1 <= bz1), 0, np.where(ty1 <= bz1, 1, 2))
        best[hit] = thi[hit]; face[hit] = f[hit]; bi[hit] = i
    return best, face, bi

STYLE = {  # base colour, material, texture
    'floor0': ((86, 80, 72), 2, 'tiles'), 'planks': ((104, 78, 54), 6, 'planks'), 'wall': ((150, 136, 116), 5, 'plaster'),
    'wallb': ((120, 104, 88), 5, 'plaster'), 'cut': ((54, 46, 40), 3, 'cut'), 'stairs': ((112, 84, 58), 6, 'steps'),
    'wood': ((96, 70, 48), 6, 'planks'), 'cloth': ((86, 92, 78), 6, 'plain'), 'metal': ((90, 94, 96), 8, 'plain'),
    'conc': ((122, 118, 110), 2, 'plaster'), 'shelf': ((80, 60, 42), 6, 'planks'),
}
def shade_layer(x0r, y0r, boxes, gloom=0.82):
    t, face, bi = box_cast(x0r, y0r, boxes)
    valid = bi >= 0
    h, w = t.shape
    col = np.zeros((h, w, 3)); mt = np.zeros((h, w), np.uint8)
    px = x0r + t; py = y0r + t; pz = t
    n = rng.normal(0, 1, (h, w))
    for i, b in enumerate(boxes):
        sel = bi == i
        if not sel.any(): continue
        base, m, tex = STYLE[b[6]]
        c = np.array(base, float)[None, :].repeat(sel.sum(), 0)
        f = face[sel]; X = px[sel]; Y = py[sel]; Z = pz[sel]
        if tex == 'planks':
            k = np.floor(Y / 2.6) if b[6] != 'shelf' else np.floor(Z / 3)
            c *= (0.86 + 0.22 * ((k * 0.618) % 1))[:, None]
            c[(Y % 2.6) < 0.35] *= 0.7
        elif tex == 'tiles':
            chk = ((np.floor(X / 4) + np.floor(Y / 4)) % 2)[:, None]; c *= 0.92 + 0.1 * chk
        elif tex == 'steps':
            c[f == 2] *= 1.12
        elif tex == 'plaster':
            c *= (0.9 + 0.12 * np.sin(X * 0.21 + Y * 0.17) * np.sin(Z * 0.33))[:, None]
        shade = np.where(f == 2, 1.0, np.where(f == 1, 0.84, 0.68))
        if b[6] == 'cut': shade = np.where(f == 2, 0.95, 0.75)
        c *= shade[:, None]
        col[sel] = c; mt[sel] = m
    col += n[..., None] * 5
    col = np.clip(col * gloom, 0, 255)
    dep = x0r + y0r + 3 * np.maximum(t, 0)
    return (col, valid, dep, np.maximum(t, 0), np.where(valid, mt, 0).astype(np.uint8))

def rect_minus(r, holes):
    """split rectangle r=(x0,y0,x1,y1) into pieces not covered by holes (each fully inside r along one axis is enough here)"""
    out = [r]
    for hx0, hy0, hx1, hy1 in holes:
        nxt = []
        for (x0, y0, x1, y1) in out:
            if hx1 <= x0 or hx0 >= x1 or hy1 <= y0 or hy0 >= y1: nxt.append((x0, y0, x1, y1)); continue
            if hx0 > x0: nxt.append((x0, y0, hx0, y1))
            if hx1 < x1: nxt.append((hx1, y0, x1, y1))
            if hy0 > y0: nxt.append((max(x0, hx0), y0, min(x1, hx1), hy0))
            if hy1 < y1: nxt.append((max(x0, hx0), hy1, min(x1, hx1), y1))
        out = nxt
    return out

def add_building(name, floors, top, door, stairs=None, props=(), roof_walk=False, stub=10, wall=3):
    o = next(ob for ob in objects if ob['name'] == name)
    X1, Y1, Wd, Dd = o['shape'][1], o['shape'][2], o['shape'][3], o['shape'][4]
    fx0, fy0 = X1 - Wd, Y1 - Dd
    ix0, iy0, ix1, iy1 = fx0 + wall, fy0 + wall, X1 - wall, Y1 - wall
    x0r, y0r = obj_grid(o)
    da, db, dh = door  # door on the front-left face (y = Y1), from x=da to x=db, height dh
    holes = []
    if stairs:
        sx0, sy0, sx1, sy1 = stairs['rect']
        holes.append((sx0, sy0, sx1, sy1))
    lflags = [None] * len(o['layers'])
    for s, fz in enumerate(floors):
        ceil = floors[s + 1] if s + 1 < len(floors) else top
        B = []
        # floor slabs of this storey and all below (seen through the stairwell)
        for i in range(s + 1):
            z = floors[i]
            for (a, b, c, d) in (rect_minus((ix0, iy0, ix1, iy1), holes) if i > 0 else [(fx0, fy0, X1, Y1)]):
                B.append((a, b, z - 2, c, d, z, 'floor0' if i == 0 else 'planks'))
        # back walls (full height to this storey's ceiling)
        B.append((fx0, fy0, 0, ix0, Y1, ceil, 'wallb'))
        B.append((fx0, fy0, 0, X1, iy0, ceil, 'wall'))
        # front walls: full for storeys below, low stub on this one; door gap
        hz = fz + stub
        for (a, b, c, d) in rect_minus((fx0, iy1, X1, Y1), [(da, iy1 - 1, db, Y1 + 1)]):
            B.append((a, b, 0, c, d, hz, 'cut'))
        if fz + stub > dh: B.append((da, iy1, dh, db, Y1, hz, 'cut'))
        B.append((ix1, fy0, 0, X1, iy1, hz, 'cut'))
        if stairs:
            lo, hi = stairs['low'], stairs['high']; n = max(2, int(np.ceil(floors[1] / 3.0)))
            L = hi[0] - lo[0]  # stairs run along x
            for k in range(1, n + 1):
                xa = lo[0] + L * (k - 1) / n
                B.append((min(xa, hi[0]), sy0, 0, max(xa, hi[0]), sy1, floors[1] * k / n, 'stairs'))
            if s >= 1:  # railing along the stairwell
                B.append((min(lo[0], hi[0]), sy1, floors[1], max(lo[0], hi[0]), sy1 + 0.8, floors[1] + 9, 'wood'))
        for p in props:
            if p['z0'] <= fz + 1 or p['z0'] < fz:
                if p['z0'] > ceil - 1: continue
                B.append((p['rect'][0], p['rect'][1], p['z0'], p['rect'][2], p['rect'][3], p['z1'], p['style']))
        o['layers'].append(shade_layer(x0r, y0r, B, gloom=0.9 if s == 0 else 0.86))
        lflags.append({'cut': s})
    # ----- door object: pixels of the door taken out of the facade, plus the open (swung inward) version -----
    rgb, m, dep, zh = o['layers'][0][:4]
    t = zh; X = x0r + t; Y = y0r + t
    sel = m & (np.abs(Y - Y1) < 1.0) & (X >= da) & (X <= db) & (t <= dh)
    ys, xs = np.nonzero(sel)
    # open version: plane x = da, y from Y1-(db-da) to Y1, hinge at x=da, outer face now facing +x
    t2 = da - x0r; Yo = y0r + t2
    op = (Yo >= Y1 - (db - da)) & (Yo <= Y1 - 0.3) & (t2 >= 0) & (t2 <= dh)
    xs_src = da + (Y1 - Yo)
    ssx = np.round((xs_src - Y1) * 2 + OX - o['x0'] - 0.5).astype(int); ssy = np.round(xs_src + Y1 - 2 * t2 + OY - o['y0'] - 0.5).astype(int)
    ok = op & (ssx >= 0) & (ssy >= 0) & (ssx < o['w']) & (ssy < o['h'])
    ok2 = ok.copy(); ok2[ok] = sel[ssy[ok], ssx[ok]]
    oys, oxs = np.nonzero(ok2)
    ay0 = min(ys.min(), oys.min()); ay1 = max(ys.max(), oys.max()) + 1; ax0 = min(xs.min(), oxs.min()); ax1 = max(xs.max(), oxs.max()) + 1
    cs = (slice(ay0, ay1), slice(ax0, ax1))
    closed = (rgb[cs].copy(), sel[cs].copy(), dep[cs].copy(), zh[cs].copy(), np.full(sel[cs].shape, 6, np.uint8))
    orgb = np.zeros_like(rgb); orgb[ok2] = rgb[ssy[ok2], ssx[ok2]] * 0.8
    odep = x0r + y0r + 3 * np.maximum(t2, 0)
    opened = (orgb[cs], ok2[cs], odep[cs], np.maximum(t2, 0)[cs], np.full(sel[cs].shape, 6, np.uint8))
    m[sel] = False  # hole in the facade
    door_name = 'Drzwi'
    objects.append(dict(name=door_name, mat=6, ex=False, x0=o['x0'] + ax0, y0=o['y0'] + ay0, w=ax1 - ax0, h=ay1 - ay0,
                        layers=[closed, opened], thick=2, shape=('box', db, Y1, db - da, 1.0, dh), blocks=False,
                        lflags=[{'door': 'closed'}, {'door': 'open', 'hide': True}], parent=name))
    o['lflags'] = lflags
    o['bld'] = dict(fp=[fx0, fy0, X1, Y1], wall=wall, floors=floors, top=round(top, 1), roofWalk=roof_walk, stub=stub,
                    door=dict(rect=[da, iy1, db, Y1], h=dh),
                    stairs=(dict(rect=list(stairs['rect']), low=list(stairs['low']), high=list(stairs['high']), z1=floors[1]) if stairs else None),
                    holes=[dict(rect=list(hh), storey=1) for hh in holes],
                    props=[dict(rect=list(p['rect']), z0=p['z0'], z1=p['z1']) for p in props])
    print(f'{name}: storeys {floors}, top {top:.1f}, door {da}-{db}, cut layers {len(floors)}')

kam_o = next(ob for ob in objects if ob['name'] == 'Kamienica')
KH = (kam_o['layers'][0][3][kam_o['layers'][0][1]]).max()
F1 = 51
add_building('Kamienica', [0, F1], 100, (71, 81, 31),
             stairs=dict(rect=(44, 18.8, 112, 31), low=(112, 25), high=(44, 25)),
             props=[dict(rect=(108, 66, 120, 78), z0=0, z1=8, style='wood'),        # table
                    dict(rect=(34.8, 52, 39, 72), z0=0, z1=20, style='shelf'),      # cupboard
                    dict(rect=(84, 84, 96, 101), z0=0, z1=6, style='cloth'),        # old sofa
                    dict(rect=(40, 80, 58, 98), z0=F1, z1=F1 + 5, style='cloth'),   # bed
                    dict(rect=(100, 60, 110, 70), z0=F1, z1=F1 + 7, style='wood'),  # crate
                    dict(rect=(34.8, 36, 39, 50), z0=F1, z1=F1 + 18, style='shelf')])
pav_o = next(ob for ob in objects if ob['name'] == 'Pawilon handlowy')
PT = pav_o['shape'][5]
add_building('Pawilon handlowy', [0], PT, (214, 233, 30), roof_walk=True,
             props=[dict(rect=(200, 62, 242, 67), z0=0, z1=9, style='wood'),       # shop counter
                    dict(rect=(185, 46, 189, 90), z0=0, z1=18, style='shelf'),      # shelves along the back wall
                    dict(rect=(192, 41.5, 250, 45), z0=0, z1=18, style='shelf'),
                    dict(rect=(246, 76, 254, 84), z0=0, z1=10, style='metal')])     # fridge

# ---------------- shop sign "STONKA": hanging lightbox perpendicular to the pavilion front ----------------
from PIL import ImageFont, ImageDraw as _ID
def sign_texture(lit):
    TW, TH = 480, 240
    im = Image.new('RGB', (TW, TH), (8, 40, 22)); d = _ID.Draw(im)
    bg = (22, 96, 48) if lit else (14, 58, 30)
    d.rounded_rectangle((6, 6, TW - 7, TH - 7), 46, fill=bg, outline=(40, 140, 70) if lit else (24, 80, 42), width=10)
    f = ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf', 92)
    txt = 'STONKA'; tw = d.textlength(txt, font=f)
    fg = (170, 255, 120) if lit else (70, 120, 60)
    d.text((112 + (TW - 132 - tw) / 2, 52), txt, font=f, fill=fg)
    # small potato beetle: yellow shell with black stripes
    bx, by = 58, 112
    d.ellipse((bx - 34, by - 44, bx + 34, by + 44), fill=(230, 190, 60) if lit else (120, 100, 40))
    for i in (-18, 0, 18): d.line((bx + i, by - 40, bx + i, by + 40), fill=(20, 20, 14), width=7)
    d.ellipse((bx - 18, by - 64, bx + 18, by - 38), fill=(200, 110, 40) if lit else (100, 60, 30))
    d.line((40, 200, TW - 40, 200), fill=fg, width=8)
    return np.asarray(im).astype(float)

def make_sign():
    pv = next(ob for ob in objects if ob['name'] == 'Pawilon handlowy')
    Y1 = pv['shape'][2]
    PX, PZ0, PZ1, L = 206.0, 19.0, 31.0, 24.0
    boxes = [(PX - 0.6, Y1, 29.4, PX + 0.6, Y1 + 2.4, 30.6, 'metal'),
             (PX - 1.0, Y1 + 1.6, PZ0, PX + 1.0, Y1 + 1.6 + L, PZ1, 'panel')]
    xs, ys = [], []
    for b in boxes:
        for x in (b[0], b[3]):
            for y in (b[1], b[4]):
                for z in (b[2], b[5]): xs.append((x - y) * 2 + OX); ys.append(x + y - 2 * z + OY)
    o = dict(name='Szyld STONKA', mat=7, ex=False, x0=int(np.floor(min(xs))) - 1, y0=int(np.floor(min(ys))) - 1)
    o['w'] = int(np.ceil(max(xs))) - o['x0'] + 2; o['h'] = int(np.ceil(max(ys))) - o['y0'] + 2
    x0r, y0r = obj_grid(o)
    t, face, bi = box_cast(x0r, y0r, [b[:6] + ('wood',) for b in boxes])
    valid = bi >= 0
    X = x0r + t; Y = y0r + t; Z = t
    layers = []
    for lit in (True, False):
        tex = sign_texture(lit); th, tw = tex.shape[:2]
        col = np.zeros(t.shape + (3,))
        pan = valid & (bi == 1)
        u = np.clip(((Y1 + 1.6 + L) - Y) / L, 0, .999); v = np.clip((PZ1 - Z) / (PZ1 - PZ0), 0, .999)
        f0 = pan & (face == 0)
        col[f0] = tex[(v[f0] * th).astype(int), (u[f0] * tw).astype(int)]
        col[pan & (face == 1)] = (150, 255, 110) if lit else (40, 90, 46)
        col[pan & (face == 2)] = (16, 70, 36)
        met = valid & (bi == 0)
        col[met] = np.array((96, 98, 100)) * np.where(face[met] == 2, 1.0, .75)[:, None]
        mt = np.where(pan, 7, 8).astype(np.uint8)
        layers.append((col, valid.copy(), x0r + y0r + 3 * np.maximum(t, 0), np.maximum(t, 0), np.where(valid, mt, 0).astype(np.uint8)))
    o.update(layers=layers, thick=2, shape=('box', PX + 1, Y1 + 1.6 + L, 2, L + 1.6, PZ1), blocks=False,
             lflags=[{'sign': 'lit'}, {'sign': 'unlit', 'hide': True, 'ghost': True}], parent='Pawilon handlowy', light=[PX + 1.2, Y1 + 1.6 + L / 2, (PZ0 + PZ1) / 2])
    objects.append(o)
make_sign()

# ---------------- ground ----------------
def tile(fname, px):
    im = Image.open(MAG + fname).convert('RGB').resize((px, px), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)
T_asf = tile('tekstura_asfalt.png', 256); T_bruk = tile('tekstura_bruk.png', 128); T_gr = tile('tekstura_trawa.png', 160)
TS = {'asf': (T_asf, 95.0), 'bruk': (T_bruk, 24.0), 'gr': (T_gr, 40.0)}
yy, xx = np.mgrid[0:BH, 0:BW]
sxp = xx + 0.5; syp = yy + 0.5
gx = ((sxp - OX) / 2 + (syp - OY)) / 2; gy = ((syp - OY) - (sxp - OX) / 2) / 2
inside = (gx >= 0) & (gy >= 0) & (gx < W) & (gy < D)
def samp(name):
    T, size = TS[name]; n = T.shape[0]
    u = (gx / size * n) % n; v = (gy / size * n) % n
    u0 = np.floor(u).astype(int) % n; v0 = np.floor(v).astype(int) % n; fu = (u - np.floor(u))[..., None]; fv = (v - np.floor(v))[..., None]
    u1 = (u0 + 1) % n; v1 = (v0 + 1) % n
    return (T[v0, u0] * (1 - fu) * (1 - fv) + T[v0, u1] * fu * (1 - fv) + T[v1, u0] * (1 - fu) * fv + T[v1, u1] * fu * fv)
def vnoise(scale, seed):
    r = np.random.default_rng(seed).random((int(W / scale) + 3, int(D / scale) + 3))
    return ndimage.map_coordinates(r, [np.clip(gx / scale, 0, r.shape[0] - 2), np.clip(gy / scale, 0, r.shape[1] - 2)], order=3, mode='nearest')
n1 = vnoise(18, 1); n2 = vnoise(6, 2)
road = (gy >= 120) & (gy < 176)
curb = ((gy >= 116) & (gy < 120)) | ((gy >= 176) & (gy < 180))
plaza = (gy >= 180) & (gx < 170 + (n1 - .5) * 30)
yard = (gy < 116)
G_ = samp('gr'); A_ = samp('asf'); Bk = samp('bruk')
col = G_.copy()
# dirt / grass variation
col *= (0.8 + 0.34 * n1)[..., None]
col = col * (1 - 0.25 * (n2 > .55)[..., None]) + (n2 > .55)[..., None] * 0.25 * col * np.array([0.92, 1.08, 0.85])
w_bruk = np.clip((plaza.astype(float) - (n2 < 0.18) * 0.8), 0, 1)
col = col * (1 - w_bruk[..., None]) + Bk * w_bruk[..., None] * (0.9 + 0.2 * n2)[..., None]
yardw = np.clip(yard * (0.55 + 0.6 * (n1 - .5)) + yard * (n2 > .6) * .3, 0, 1)
col = col * (1 - yardw[..., None]) + (Bk * 0.55 + G_ * 0.45) * yardw[..., None] * 0.92
col[road] = A_[road] * (0.92 + 0.16 * n1[road])[..., None]
col[curb] = (np.array([150, 146, 138]) * (0.9 + 0.2 * n2[curb])[..., None])
# lane marking (worn dashes)
mark = road & (np.abs(gy - 148) < 1.2) & (((gx // 20) % 2) == 0) & (n2 > 0.3)
col[mark] = col[mark] * 0.4 + np.array([190, 178, 120]) * 0.6
col *= 1.02
out = np.zeros((BH, BW, 4), np.uint8)
out[..., :3] = np.clip(col, 0, 255); out[..., 3] = np.where(inside, 255, 0)
# slab sides (diorama edge)
for side in range(1, 14):
    sh = np.zeros_like(inside); sh[side:] = inside[:-side]
    edge = sh & ~inside & (out[..., 3] == 0)
    shade = 70 - side * 2
    out[edge, :3] = [shade + 10, shade, shade - 12]; out[edge, 3] = 255
Image.fromarray(out).save(OUT + 'ground.png', optimize=True)

# ---------------- pack objects ----------------
def hsv(rgb):
    r, g, b = rgb[..., 0] / 255, rgb[..., 1] / 255, rgb[..., 2] / 255
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b); d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, np.where(mx > 0, d / (mx + 1e-6), 0), mx
# materials: 1 soil 2 concrete 3 brick 4 stone 5 plaster 6 wood 7 glass 8 metal 9 tile 10 foliage 11 sand
def classify(name, rgb, zh, mask):
    h, sat, v = hsv(rgb); r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mt = np.full(mask.shape, 2, np.uint8)
    brickish = ((h < 22) | (h > 340)) & (sat > .32) & (v > .2)
    brown = (h >= 16) & (h < 50) & (sat > .25) & (v < .58)
    darkgl = (v < .27) & (sat < .38)
    bluegl = (b > r + 6) & (sat < .4) & (v > .3)
    green = (h > 45) & (h < 175) & (sat > .16)
    if name == 'Kamienica':
        eave = np.percentile(zh[mask], 55)
        roof = zh > eave
        mt[:] = 5; mt[brickish | brown] = 3
        mt[roof] = 9
        mt[roof & (sat < .5) & (h > 15) & (h < 55) & (v < .62)] = 6
        mt[roof & (sat < .18)] = 5
        mt[(darkgl | bluegl) & ~roof] = 7
        mt[(h > 180) & (h < 250) & (sat < .3) & (v > .3) & (v < .62) & ~bluegl] = 8
    elif name == 'Pawilon handlowy':
        low = zh < np.percentile(zh[mask], 60); mt[:] = 2; mt[(darkgl | bluegl) & low] = 7; mt[(h > 8) & (h < 40) & (sat > .35) & (v < .6) & low] = 8
    elif name == 'Wrak samochodu':
        mt[:] = 8; mt[(bluegl | (darkgl & (v > .1))) & (zh > np.percentile(zh[mask], 45))] = 7
    elif name in ('Beczka', 'Beczka z paliwem', 'Latarnia'): mt[:] = 8
    if name == 'Latarnia':
        gl = np.zeros_like(mask); gl[36:48, 5:20] = True; mt[gl & mask & (v > .25)] = 7
    elif name == 'Worki z piaskiem': mt[:] = 11
    elif name == 'Bariera betonowa': mt[:] = 2
    elif name == 'Skrzynie': mt[:] = 6
    elif name in ('Trawa', 'Krzak'): mt[:] = 10
    elif name == 'Gruz': mt[:] = 2; mt[brickish] = 3
    elif name in ('Sosna', 'Brzoza'):
        mt[:] = 6; mt[green | ((h > 28) & (h < 75) & (sat > .42) & (v > .42))] = 10
    return mt
tiles = []
meta = []
for o in objects:
    L = []
    for li, lay in enumerate(o['layers']):
        rgb, m, dep, zh = lay[:4]
        mtl = lay[4] if len(lay) > 4 else classify(o['name'], rgb, zh, m)
        hgt, wid = m.shape
        c = np.zeros((hgt, wid, 4), np.uint8); c[..., :3] = np.clip(rgb, 0, 255); c[..., 3] = np.where(m, 255, 0)
        g = np.zeros((hgt, wid, 3), np.uint8)
        d16 = np.clip(np.round(dep * 16), 0, 65535).astype(np.int64)
        g[..., 0] = d16 >> 8; g[..., 1] = d16 & 255; g[..., 2] = np.clip(np.round(zh), 0, 255)
        tiles.append((c, g, np.where(m, mtl, 0).astype(np.uint8))); L.append(len(tiles) - 1)
    s = o['shape']
    if s[0] == 'box': fp = [s[1] - s[3], s[2] - s[4], s[1], s[2]]
    else: fp = [s[1] - s[3], s[2] - s[3], s[1] + s[3], s[2] + s[3]]
    # centre (for barrel explosions etc.)
    rgb, m, dep, zh = o['layers'][0][:4]
    ys_, xs_ = np.nonzero(m); dd = dep[m].mean(); zz = zh[m].mean(); sxm = xs_.mean() + o['x0']
    df = (sxm - OX) / 2; ss = dd - zz
    meta.append(dict(name=o['name'], mat=o['mat'], ex=o['ex'], x0=o['x0'], y0=o['y0'], w=o['w'], h=o['h'], layers=L, thick=o['thick'],
                     fp=[round(v, 1) for v in fp], blocks=o['blocks'], topple=o['name'] in ('Sosna', 'Brzoza', 'Latarnia'), c=[round((ss + df) / 2, 1), round((ss - df) / 2, 1), round(zz, 1)]))
    for k in ('lflags', 'bld', 'parent', 'light'):
        if o.get(k) is not None: meta[-1][k] = o[k]
AW = 2048; x = y = rowh = 0; pos = {}
for i in sorted(range(len(tiles)), key=lambda i: -tiles[i][0].shape[0]):
    hgt, wid = tiles[i][0].shape[:2]
    if x + wid > AW: x = 0; y += rowh + 1; rowh = 0
    pos[i] = (x, y); x += wid + 1; rowh = max(rowh, hgt)
AH = y + rowh + 1
CA = np.zeros((AH, AW, 4), np.uint8); GA = np.zeros((AH, AW, 3), np.uint8); MA = np.zeros((AH, AW), np.uint8)
for i, (c, g, mm) in enumerate(tiles):
    px_, py_ = pos[i]; hgt, wid = c.shape[:2]; CA[py_:py_ + hgt, px_:px_ + wid] = c; GA[py_:py_ + hgt, px_:px_ + wid] = g; MA[py_:py_ + hgt, px_:px_ + wid] = mm
Image.fromarray(CA).save(OUT + 'world_color.png', optimize=True)
Image.fromarray(GA).save(OUT + 'world_geo.png', optimize=True)
Image.fromarray(MA, 'L').convert('RGB').save(OUT + 'world_mat.png', optimize=True)
json.dump(dict(W=W, D=D, H=H, objects=meta, rects=[[pos[i][0], pos[i][1], tiles[i][0].shape[1], tiles[i][0].shape[0]] for i in range(len(tiles))]),
          open(OUT + 'world.json', 'w'), separators=(',', ':'), default=lambda v: v.item() if hasattr(v, 'item') else str(v))
print('objects', len(objects), 'atlas', AW, AH)
# preview composite (painter by depth) for a quick look
prev = out.astype(np.float32).copy(); zb = np.full((BH, BW), -1e9)
yyg = np.arange(BH)[:, None] - OY + 0.0; zb[:] = np.where(inside, yyg + 0 * xx, -1e9)
for o in objects:
    for lay in o['layers']:
        rgb, m, dep, zh = lay[:4]
        hgt, wid = m.shape; ys_, xs_ = np.nonzero(m)
        Y = ys_ + o['y0']; X = xs_ + o['x0']; ok = (Y >= 0) & (Y < BH) & (X >= 0) & (X < BW)
        Y, X, yy_, xx_ = Y[ok], X[ok], ys_[ok], xs_[ok]
        d = dep[yy_, xx_]; win = d > zb[Y, X]
        zb[Y[win], X[win]] = d[win]; prev[Y[win], X[win], :3] = rgb[yy_[win], xx_[win]]; prev[Y[win], X[win], 3] = 255
Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8)).save('world_preview.png')
