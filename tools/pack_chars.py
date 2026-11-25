import json, sys, numpy as np
from PIL import Image
AX, AY = 64, 100
def srgb2lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
out_meta = {}
tiles = []  # (key, color rgba uint8, geo rgba uint8)
for key, kind in (('ty', 'human'), ('zlom', 'human'), ('pies', 'dog')):
    meta = json.load(open(f'out_{key}/meta.json'))
    parts = meta['parts']; frames = []; err = []
    for d in range(8):
        for pose in range(7):
            fr = []
            for p in parts:
                ci = np.asarray(Image.open(f'out_{key}/c_{pose}_{d}_{p}.png')); c = ci.astype(np.float64) / (65535.0 if ci.dtype == np.uint16 else 255.0)
                gi = np.asarray(Image.open(f'out_{key}/p_{pose}_{d}_{p}.png')); g = gi.astype(np.float64) / (65535.0 if gi.dtype == np.uint16 else 255.0)
                m = (c[..., 3] > 0.5) & (g[..., 3] > 0.5)
                if not m.any(): fr.append(None); continue
                ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                gl = srgb2lin(g[..., :3])
                dep = gl[..., 0] * 128 - 64; zh = gl[..., 1] * 64
                # consistency check: sy = (x+y) - 2z + AY ; depth = x+y+z  ->  sy = dep - 3z + AY
                sy_pred = dep - 3 * zh + AY
                err.append(np.abs(sy_pred[m] - (ys + 0.5)).mean())
                col = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
                a = np.clip(c[..., 3:4], 1e-4, 1)
                rgb = np.clip(c[..., :3] / a, 0, 1)  # un-premultiply (straight already but safe)
                rgb = np.clip(c[..., :3], 0, 1) if c[..., :3].max() <= 1 else rgb
                col[..., :3] = (rgb[y0:y1, x0:x1] * 255 + .5).astype(np.uint8)
                col[..., 3] = np.where(m[y0:y1, x0:x1], 255, 0)
                geo = np.zeros_like(col)
                geo[..., 0] = np.clip(np.round((dep[y0:y1, x0:x1] + 64) * 2), 0, 255)
                geo[..., 1] = np.clip(np.round(zh[y0:y1, x0:x1] * 4), 0, 255)
                geo[..., 2] = 255; geo[..., 3] = col[..., 3]
                tiles.append((col, geo)); fr.append([len(tiles) - 1, int(x0 - AX), int(y0 - AY)])
            frames.append(fr)
    print(key, 'mean sy err px', round(float(np.mean(err)), 3), 'max', round(float(np.max(err)), 3))
    out_meta[key] = {'parts': parts, 'frames': frames, 'joints': meta['poses']}
# shelf pack
W = 1024; x = y = rowh = 0; rects = []
order = sorted(range(len(tiles)), key=lambda i: -tiles[i][0].shape[0])
pos = {}
for i in order:
    h, w = tiles[i][0].shape[:2]
    if x + w > W: x = 0; y += rowh + 1; rowh = 0
    pos[i] = (x, y); x += w + 1; rowh = max(rowh, h)
H = y + rowh + 1
A = np.zeros((H, W, 4), np.uint8); B = np.zeros((H, W, 4), np.uint8)
for i, (c, g) in enumerate(tiles):
    px, py = pos[i]; h, w = c.shape[:2]; A[py:py + h, px:px + w] = c; B[py:py + h, px:px + w] = g
Image.fromarray(A).save('../assets/chars_color.png', optimize=True)
Image.fromarray(B).save('../assets/chars_geo.png', optimize=True)
rect = [[pos[i][0], pos[i][1], tiles[i][0].shape[1], tiles[i][0].shape[0]] for i in range(len(tiles))]
json.dump({'rects': rect, 'types': out_meta}, open('../assets/chars.json', 'w'), separators=(',', ':'))
print('atlas', W, H, 'tiles', len(tiles))
