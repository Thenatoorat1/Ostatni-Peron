# Packs the character atlas: the new detailed player (arm sets + clothing/weapon overlays, rendered at 2x and
# downsampled here) and the old enemy/dog sprites.
import json, os, numpy as np
from PIL import Image
AX, AY = 64, 100
def srgb2lin(v): return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
def load(path):
    a = np.asarray(Image.open(path)); return a.astype(np.float64) / (65535.0 if a.dtype == np.uint16 else 255.0)
tiles = []
def add_tile(c, g, m, ss, shift=(0.0, 0.0)):
    """c,g: float RGBA arrays at render scale ss; m: valid mask. Returns [tile, ox, oy] or None"""
    if not m.any(): return None
    gl = srgb2lin(g[..., :3]); dep = gl[..., 0] * 128 - 64; zh = gl[..., 1] * 64
    h, w = m.shape
    if ss > 1:
        H, W_ = h // ss, w // ss
        def blk(a): return a[:H * ss, :W_ * ss].reshape(H, ss, W_, ss, *a.shape[2:])
        al = blk(c[..., 3]).mean(axis=(1, 3))
        pre = blk(c[..., :3] * c[..., 3:4]).sum(axis=(1, 3)); asum = blk(c[..., 3]).sum(axis=(1, 3))
        rgb = pre / np.maximum(asum, 1e-6)[..., None]
        mv = blk(m.astype(float)); cnt = mv.sum(axis=(1, 3))
        dd = (blk(dep) * mv).sum(axis=(1, 3)) / np.maximum(cnt, 1); zz = (blk(zh) * mv).sum(axis=(1, 3)) / np.maximum(cnt, 1)
        mm = (al >= 0.5) & (cnt > 0)
        # the downsampled pixel sits at the block centre: keep sy = dep - 3z + AY consistent
        yy = np.arange(H)[:, None] + 0.5
        dd = np.where(mm, dd, 0); zz = np.where(mm, zz, 0)
        err = np.abs(dd - 3 * zz + AY - yy)[mm].mean() if mm.any() else 0
        if err > 2: dd = np.where(mm, yy - AY + 3 * zz, 0)
        m, dep, zh = mm, dd, zz
    else:
        rgb = c[..., :3]
    if not m.any(): return None
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    col = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    col[..., :3] = (np.clip(rgb[y0:y1, x0:x1], 0, 1) * 255 + .5).astype(np.uint8); col[..., 3] = np.where(m[y0:y1, x0:x1], 255, 0)
    geo = np.zeros_like(col)
    geo[..., 0] = np.clip(np.round((dep[y0:y1, x0:x1] + 64) * 2), 0, 255); geo[..., 1] = np.clip(np.round(zh[y0:y1, x0:x1] * 4), 0, 255)
    geo[..., 2] = 255; geo[..., 3] = col[..., 3]
    if shift != (0.0, 0.0):
        wx, wy = shift; dsx = round(2 * (wx - wy)); dsy = round(wx + wy)
        geo[..., 0] = np.where(col[..., 3] > 0, np.clip(geo[..., 0].astype(int) - round((wx + wy) * 2), 0, 255), 0)
        return_ox, return_oy = int(x0 - AX) - dsx, int(y0 - AY) - dsy
        tiles.append((col, geo)); return [len(tiles) - 1, return_ox, return_oy]
    tiles.append((col, geo)); return [len(tiles) - 1, int(x0 - AX), int(y0 - AY)]

out = {}
# ---- old enemies and dog ----
for key in ('zlom', 'pies'):
    meta = json.load(open(f'out2_{key}/meta.json')); parts = meta['parts']; frames = []
    for d in range(8):
        for pose in range(len(meta['poseNames'])):
            fr = []
            for p in parts:
                c = load(f'out2_{key}/c_{pose}_{d}_{p}.png'); g = load(f'out2_{key}/p_{pose}_{d}_{p}.png')
                fr.append(add_tile(c, g, (c[..., 3] > .5) & (g[..., 3] > .5), 1))
            frames.append(fr)
    out[key] = {'parts': parts, 'frames': frames, 'joints': [meta['poses'][str(i)]['joints'] for i in range(len(meta['poseNames']))], 'poseNames': meta['poseNames']}
    print(key, len(tiles))
# ---- new player ----
SRC = 'out3_ty'
M = {}
for f in os.listdir(SRC):
    if f.startswith('meta_'):
        mm = json.load(open(f'{SRC}/{f}'))
        M.setdefault('poseNames', mm['poseNames']); M.setdefault('sets', {}).update(mm['sets'])
NPOSE = len(M['poseNames']); sets = {}
import math
OFF = (0.0, 0.0)  # sprites are rendered with the hips on the anchor
def dshift(d):
    a = d * math.pi / 4; c, s_ = math.cos(a), math.sin(a)
    return (OFF[0] * c - OFF[1] * s_, OFF[0] * s_ + OFF[1] * c)
print('model offset', OFF)
for st, sd in M['sets'].items():
    names = list(sd['parts'].keys()); frames = [None] * (8 * NPOSE)
    for d in range(8):
        for pose in sd['poses']:
            fr = []
            for n in names:
                cp = f'{SRC}/{st}_c_{pose}_{d}_{n}.png'
                if not os.path.exists(cp): fr.append(None); continue
                c = load(cp); g = load(f'{SRC}/{st}_p_{pose}_{d}_{n}.png')
                fr.append(add_tile(c, g, (c[..., 3] > .5) & (g[..., 3] > .5), 2, dshift(d)))
            frames[d * NPOSE + pose] = fr
    jo = {int(k): {jn: [round(jv[0] - OFF[0], 2), round(jv[1] - OFF[1], 2), jv[2]] for jn, jv in J.items()} for k, J in sd['joints'].items()}
    sets[st] = {'parts': names, 'base': sd['parts'], 'poses': sd['poses'], 'frames': frames, 'joints': {str(k): v for k, v in jo.items()}}
    print(st, len(tiles))
out['ty'] = {'sets': sets, 'poseNames': M['poseNames'], 'parts': ['legL', 'legR', 'torso', 'armL', 'armR', 'head']}
# ---- shelf pack ----
W = 2048; x = y = rowh = 0; pos = {}
for i in sorted(range(len(tiles)), key=lambda i: -tiles[i][0].shape[0]):
    h, w = tiles[i][0].shape[:2]
    if x + w > W: x = 0; y += rowh + 1; rowh = 0
    pos[i] = (x, y); x += w + 1; rowh = max(rowh, h)
H = y + rowh + 1
A = np.zeros((H, W, 4), np.uint8); B = np.zeros((H, W, 4), np.uint8)
for i, (c, g) in enumerate(tiles):
    px, py = pos[i]; h, w = c.shape[:2]; A[py:py + h, px:px + w] = c; B[py:py + h, px:px + w] = g
Image.fromarray(A).save('../real/assets/chars_color.png', optimize=True)
Image.fromarray(B).save('../real/assets/chars_geo.png', optimize=True)
json.dump({'rects': [[pos[i][0], pos[i][1], tiles[i][0].shape[1], tiles[i][0].shape[0]] for i in range(len(tiles))], 'types': out}, open('../real/assets/chars.json', 'w'), separators=(',', ':'))
print('atlas', W, H, 'tiles', len(tiles))
