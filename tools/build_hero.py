# Character portrait for the inventory window: painted base figure + clothing layers cut out of aligned edits.
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
def load(f): return np.asarray(Image.open(f'mag6/{f}.png').convert('RGB')).astype(np.float32)
def key(a):
    m = np.minimum(a[..., 0], a[..., 2]) - a[..., 1]
    al = 1 - np.clip((m - 55) / 50, 0, 1); al[a[..., 1] > 140] = 1
    sp = np.clip(m - 10, 0, None) * (1 - al * 0.6); a = a.copy(); a[..., 0] -= sp * .55; a[..., 2] -= sp * .55
    return np.clip(a, 0, 255), al
base = load('base'); brgb, bal = key(base)
H = 640; sc = H / base.shape[0]
ys, xs = np.nonzero(bal > .5); x0, x1 = xs.min() - 40, xs.max() + 40; y0, y1 = max(0, ys.min() - 20), min(base.shape[0], ys.max() + 20)
def save(rgb, al, name):
    im = Image.fromarray(np.dstack([rgb, al * 255]).astype(np.uint8), 'RGBA').crop((x0, y0, x1, y1))
    im = im.resize((round(im.width * H / (y1 - y0)), H), Image.LANCZOS); im.save(f'real/assets/hero_{name}.webp', quality=88, method=6); return im.size
print('base', save(brgb, bal, 'base'))
for name in ['jacket', 'rain', 'gloves', 'hat']:
    v = load(name); vrgb, val = key(v)
    d = np.sqrt(((v - base) ** 2).sum(-1))
    m = (d > 38) & (val > .5)
    m = ndimage.binary_opening(m, iterations=2); m = ndimage.binary_closing(m, iterations=4)
    lab, n = ndimage.label(m); sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 1500]); keep = ndimage.binary_fill_holes(keep)
    a = ndimage.gaussian_filter(ndimage.binary_dilation(keep, iterations=2).astype(np.float32), 1.6) * val
    print(name, n, save(vrgb, np.clip(a, 0, 1), name))
