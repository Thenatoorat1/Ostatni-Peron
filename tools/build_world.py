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
        return (col, valid, dep, z)
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
for (cx, cy, hh) in [(292, 40, 300), (286, 120, 270), (20, 130, 280), (14, 40, 300), (180, 40, 260), (60, 20, 250)]:
    place_cyl_img(pine, (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1), hh / ph, 'Sosna', 5, cx, cy, 2.2)
for (cx, cy, hh) in [(40, 250, 200), (282, 264, 180)]:
    place_cyl_img(birch, (ys2.min(), ys2.max() + 1, xs2.min(), xs2.max() + 1), hh / bh, 'Brzoza', 5, cx, cy, 1.6)

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
tiles = []
meta = []
for o in objects:
    L = []
    for (rgb, m, dep, zh) in o['layers']:
        hgt, wid = m.shape
        c = np.zeros((hgt, wid, 4), np.uint8); c[..., :3] = np.clip(rgb, 0, 255); c[..., 3] = np.where(m, 255, 0)
        g = np.zeros((hgt, wid, 3), np.uint8)
        d16 = np.clip(np.round(dep * 16), 0, 65535).astype(np.int64)
        g[..., 0] = d16 >> 8; g[..., 1] = d16 & 255; g[..., 2] = np.clip(np.round(zh), 0, 255)
        tiles.append((c, g)); L.append(len(tiles) - 1)
    s = o['shape']
    if s[0] == 'box': fp = [s[1] - s[3], s[2] - s[4], s[1], s[2]]
    else: fp = [s[1] - s[3], s[2] - s[3], s[1] + s[3], s[2] + s[3]]
    # centre (for barrel explosions etc.)
    rgb, m, dep, zh = o['layers'][0]
    ys_, xs_ = np.nonzero(m); dd = dep[m].mean(); zz = zh[m].mean(); sxm = xs_.mean() + o['x0']
    df = (sxm - OX) / 2; ss = dd - zz
    meta.append(dict(name=o['name'], mat=o['mat'], ex=o['ex'], x0=o['x0'], y0=o['y0'], w=o['w'], h=o['h'], layers=L, thick=o['thick'],
                     fp=[round(v, 1) for v in fp], blocks=o['blocks'], c=[round((ss + df) / 2, 1), round((ss - df) / 2, 1), round(zz, 1)]))
AW = 2048; x = y = rowh = 0; pos = {}
for i in sorted(range(len(tiles)), key=lambda i: -tiles[i][0].shape[0]):
    hgt, wid = tiles[i][0].shape[:2]
    if x + wid > AW: x = 0; y += rowh + 1; rowh = 0
    pos[i] = (x, y); x += wid + 1; rowh = max(rowh, hgt)
AH = y + rowh + 1
CA = np.zeros((AH, AW, 4), np.uint8); GA = np.zeros((AH, AW, 3), np.uint8)
for i, (c, g) in enumerate(tiles):
    px_, py_ = pos[i]; hgt, wid = c.shape[:2]; CA[py_:py_ + hgt, px_:px_ + wid] = c; GA[py_:py_ + hgt, px_:px_ + wid] = g
Image.fromarray(CA).save(OUT + 'world_color.png', optimize=True)
Image.fromarray(GA).save(OUT + 'world_geo.png', optimize=True)
json.dump(dict(W=W, D=D, H=H, objects=meta, rects=[[pos[i][0], pos[i][1], tiles[i][0].shape[1], tiles[i][0].shape[0]] for i in range(len(tiles))]),
          open(OUT + 'world.json', 'w'), separators=(',', ':'))
print('objects', len(objects), 'atlas', AW, AH)
# preview composite (painter by depth) for a quick look
prev = out.astype(np.float32).copy(); zb = np.full((BH, BW), -1e9)
yyg = np.arange(BH)[:, None] - OY + 0.0; zb[:] = np.where(inside, yyg + 0 * xx, -1e9)
for o in objects:
    for (rgb, m, dep, zh) in o['layers']:
        hgt, wid = m.shape; ys_, xs_ = np.nonzero(m)
        Y = ys_ + o['y0']; X = xs_ + o['x0']; ok = (Y >= 0) & (Y < BH) & (X >= 0) & (X < BW)
        Y, X, yy_, xx_ = Y[ok], X[ok], ys_[ok], xs_[ok]
        d = dep[yy_, xx_]; win = d > zb[Y, X]
        zb[Y[win], X[win]] = d[win]; prev[Y[win], X[win], :3] = rgb[yy_[win], xx_[win]]; prev[Y[win], X[win], 3] = 255
Image.fromarray(np.clip(prev, 0, 255).astype(np.uint8)).save('world_preview.png')
