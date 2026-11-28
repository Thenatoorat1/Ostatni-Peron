# Packs inventory icons (Do Zmroku atlases, Magnific clothing sheet, procedural grenade/charge) into one atlas.
import json, numpy as np
from PIL import Image, ImageDraw, ImageFilter
CELL = 48
D = 'dz/'
items_at = Image.open(D + 'icons_items.webp').convert('RGBA')
food_at = Image.open(D + 'icons_food.webp').convert('RGBA')
dz = json.load(open('/tmp/dz_items.json'))
SRC = {}
for i in dz['items']['items']: SRC[i['id']] = ('items', i['atlas'], i['w'], i['h'])
for i in dz['food']['items']: SRC.setdefault(i['id'], ('food', i['atlas'], i['w'], i['h']))

def fit(im, w, h):
    """contain-fit an RGBA image into w*CELL x h*CELL with a small margin"""
    bb = im.getbbox() or (0, 0, im.width, im.height); im = im.crop(bb)
    W, H = w * CELL - 4, h * CELL - 4; s = min(W / im.width, H / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    out = Image.new('RGBA', (w * CELL, h * CELL)); out.alpha_composite(im, ((w * CELL - im.width) // 2, (h * CELL - im.height) // 2)); return out

def from_atlas(id_):
    src, a, w, h = SRC[id_]
    at = items_at if src == 'items' else food_at
    return at.crop((a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h'])), w, h

def key_magenta(im):
    a = np.asarray(im.convert('RGB')).astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mag = (r > 150) & (b > 150) & (g < 120) & (np.abs(r - b) < 90)
    al = np.where(mag, 0, 255).astype(np.uint8)
    o = np.dstack([a.astype(np.uint8), al]); img = Image.fromarray(o, 'RGBA')
    # trim magenta fringe
    al2 = Image.fromarray(al).filter(ImageFilter.MinFilter(3)); img.putalpha(al2); return img

clothes = key_magenta(Image.open('mag2/ubrania.png'))
q = clothes.width // 2
CL = {'kurtka': clothes.crop((0, 0, q, q)), 'czapka': clothes.crop((q, 0, 2 * q, q)), 'rekawice': clothes.crop((0, q, q, 2 * q)), 'peleryna': clothes.crop((q, q, 2 * q, 2 * q))}

def grenade():
    S = 256; im = Image.new('RGBA', (S, S)); d = ImageDraw.Draw(im)
    d.ellipse((50, 80, 200, 236), fill=(74, 86, 52, 255))
    for i in range(5):
        d.line((50, 100 + i * 27, 200, 100 + i * 27), fill=(52, 62, 36, 255), width=5)
    for i in range(5):
        d.line((62 + i * 32, 82, 62 + i * 32, 234), fill=(52, 62, 36, 255), width=5)
    d.ellipse((80, 100, 120, 140), fill=(110, 124, 82, 255))
    d.rectangle((100, 44, 150, 86), fill=(120, 122, 118, 255)); d.polygon([(150, 50), (214, 70), (214, 150), (200, 150), (200, 80), (150, 72)], fill=(150, 152, 146, 255))
    d.ellipse((60, 20, 112, 72), outline=(170, 170, 160, 255), width=8)
    return im

def charge():
    S = 256; im = Image.new('RGBA', (S * 2, S)); d = ImageDraw.Draw(im)
    for i in range(3):
        y = 50 + i * 54; d.rounded_rectangle((40, y, 440, y + 50), 20, fill=(176, 46, 36, 255)); d.rectangle((40, y + 6, 440, y + 14), fill=(206, 80, 64, 255))
    for x in (120, 330): d.rectangle((x, 40, x + 34, 216), fill=(150, 150, 140, 255))
    d.rounded_rectangle((190, 76, 300, 160), 8, fill=(40, 42, 40, 255)); d.rectangle((204, 92, 286, 128), fill=(120, 20, 18, 255))
    d.line((300, 110, 380, 30), fill=(200, 40, 30, 255), width=6); d.line((300, 130, 400, 40), fill=(40, 90, 200, 255), width=6)
    return im

# id: (source, w, h)
LIST = {
 'bandage': 'dz', 'first_aid_kit': 'dz', 'painkillers': 'dz', 'antibiotics': 'dz', 'disinfectant': 'dz',
 'water_bottle': 'dz', 'canteen': 'dz', 'canned_food': 'dz', 'energy_bar': 'dz', 'chocolate': 'dz', 'canned_meat': 'dz',
 'sardines': 'dz', 'crackers': 'dz', 'mre': 'dz', 'vodka': 'dz', 'soda_can': 'dz', 'canned_soup': 'dz', 'tea_bags': 'dz',
 'scrap_metal': 'dz', 'duct_tape': 'dz', 'rope': 'dz', 'rags': 'dz', 'batteries': 'dz', 'nuts_bolts': 'dz', 'glue': 'dz',
 'fuel_can': 'dz', 'copper_wire': 'dz',
 'kitchen_knife': 'dz', 'hunting_knife': 'dz', 'crowbar': 'dz', 'hatchet': 'dz', 'machete': 'dz', 'baseball_bat': 'dz',
 'steel_pipe': 'dz', 'pipe_wrench': 'dz', 'police_baton': 'dz',
 'family_photo': 'dz', 'walkie_talkie': 'dz', 'dog_tags': 'dz', 'map': 'dz', 'locket': 'dz',
}
icons = {}
for k in LIST: im, w, h = from_atlas(k); icons[k] = (fit(im, w, h), w, h)
icons['ammo_762'] = (fit(Image.open(D + 'icons_ammo_naboje_762.webp').convert('RGBA'), 1, 1), 1, 1)
icons['ammo_shell'] = (fit(Image.open(D + 'icons_ammo_srut.webp').convert('RGBA'), 1, 1), 1, 1)
icons['rifle'] = (fit(Image.open(D + 'snajperka.webp').convert('RGBA'), 5, 1), 5, 1)
icons['shotgun'] = (fit(Image.open(D + 'strzelba.webp').convert('RGBA'), 4, 1), 4, 1)
icons['molotov'] = (fit(Image.open(D + 'molotov.webp').convert('RGBA'), 1, 2), 1, 2)
icons['grenade'] = (fit(grenade(), 1, 1), 1, 1)
icons['charge'] = (fit(charge(), 2, 1), 2, 1)
icons['jacket'] = (fit(CL['kurtka'], 2, 3), 2, 3)
icons['hat'] = (fit(CL['czapka'], 2, 2), 2, 2)
icons['gloves'] = (fit(CL['rekawice'], 2, 2), 2, 2)
icons['raincoat'] = (fit(CL['peleryna'], 2, 2), 2, 2)

# simple shelf packing
order = sorted(icons, key=lambda k: -icons[k][0].height)
AW = 16 * CELL; x = y = rowh = 0; rects = {}
for k in order:
    im = icons[k][0]
    if x + im.width > AW: x = 0; y += rowh; rowh = 0
    rects[k] = [x, y, im.width, im.height, icons[k][1], icons[k][2]]; x += im.width; rowh = max(rowh, im.height)
atlas = Image.new('RGBA', (AW, y + rowh))
for k, r in rects.items(): atlas.alpha_composite(icons[k][0], (r[0], r[1]))
atlas.save('real/assets/items.png', optimize=True)
json.dump({'cell': CELL, 'rects': rects}, open('real/assets/items.json', 'w'))
print(atlas.size, len(rects))
