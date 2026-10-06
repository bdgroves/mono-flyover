"""
MONO: a forge3d flight from Tioga Pass down Lee Vining Canyon, out over Mono
Lake and its islands, past the South Tufa towers and along the Mono Craters,
on a late-September morning.

    python tools/render.py path    --prep prep --out renders          # flight map + timing, no rendering
    python tools/render.py stills  --prep prep --out renders          # a few frames along the flight
    python tools/render.py flyover --prep prep --frames frames --chunk 3 --chunks 60
    python tools/render.py flyover --prep prep --frames frames --chunks 60 --all   # one machine
    python tools/render.py encode  --frames frames --out renders

Terrain: USGS 3DEP 1 m lidar near the camera, in nested layers (layers.py).
Colour: USDA NAIP aerial photography (0.6 m near the camera); Sentinel-2 for
the far skyline. The Sun is placed with forge3d's solar calculator. Runs
headless on a CPU (Mesa llvmpipe + Xvfb) or on a GPU.

Viewer world coordinates (from the forge3d source): x = UTM easting,
y = elevation - DEM minimum, z = -UTM northing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.warp import transform
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONTS = HERE / "fonts"
CRS = "EPSG:26911"
LAT0, LON0 = 37.955, -119.128          # Lee Vining: reference point for the Sun


class _Pacific(tzinfo):
    """US Pacific Time with daylight saving, for Pythons with no tz database."""
    def _dst(self, dt):
        from datetime import date
        y = dt.year
        mar = date(y, 3, 8) + timedelta(days=(6 - date(y, 3, 8).weekday()) % 7)
        nov = date(y, 11, 1) + timedelta(days=(6 - date(y, 11, 1).weekday()) % 7)
        n = dt.replace(tzinfo=None)
        return datetime(mar.year, mar.month, mar.day, 2) <= n < datetime(nov.year, nov.month, nov.day, 1)

    def utcoffset(self, dt):
        return timedelta(hours=-7 if self._dst(dt) else -8)

    def dst(self, dt):
        return timedelta(hours=1 if self._dst(dt) else 0)

    def tzname(self, dt):
        return "PDT" if self._dst(dt) else "PST"

    def fromutc(self, dt):
        st = dt + timedelta(hours=-8)
        return (dt + timedelta(hours=-7) if self._dst(st) else st).replace(tzinfo=self)


try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    TZ = _Pacific()

# The morning of the flight: late September, mid-morning, the Sun up in the
# east-south-east lighting the Sierra's east face and throwing the craters'
# shadows west.
WHEN = datetime(2026, 9, 26, 8, 45, tzinfo=TZ)


# ── scene ─────────────────────────────────────────────────────────────────────
class Scene:
    def __init__(self, prep: Path, work: Path, source: str = "usgs"):
        self.crs = CRS
        self.source = source
        self.work = work
        self.dem_src = prep / "mono_dem_8m.tif"
        with rasterio.open(self.dem_src) as s:
            self.dem = s.read(1)
            self.T = s.transform
            self.B = s.bounds
            self.res = s.res[0]
        self.min_h = float(np.nanmin(self.dem))
        self.naip_src = prep / "mono_naip_5m.tif"
        with rasterio.open(self.naip_src) as s:
            self.naip_bounds = s.bounds
        self.horizon_src = prep / "horizon_dem_60m.tif"
        self.horizon_tex_src = prep / "horizon_s2_60m.tif"
        self.texture = work / "naip_8k_graded.png"
        if not self.texture.exists():
            grade_texture(self.naip_src, self.texture, self.dem_src)
        self.places = {p["id"]: p for p in json.loads((ROOT / "data/places.json").read_text())["places"]}
        from layers import far_layer, horizon_layer
        self.far = far_layer(self, work)
        self.horizon = horizon_layer(self, work)
        self._gmax = {}

    def utm(self, lon, lat):
        (x,), (y,) = transform("EPSG:4326", CRS, [lon], [lat])
        return x, y

    def ground(self, x, y):
        c, r = ~self.T * (x, y)
        return float(ndimage.map_coordinates(self.dem, [[r - 0.5], [c - 0.5]], order=1, mode="nearest")[0])

    def ground_max_grid(self, radius):
        if radius not in self._gmax:
            k = max(1, int(round(radius / self.res)))
            self._gmax[radius] = ndimage.maximum_filter(self.dem, size=2 * k + 1)
        return self._gmax[radius]

    def ground_max(self, x, y, radius=150.0):
        c, r = ~self.T * (x, y)
        g = self.ground_max_grid(radius)
        return float(ndimage.map_coordinates(g, [[r - 0.5], [c - 0.5]], order=1, mode="nearest")[0])

    def world(self, x, y, h):
        return np.array([x, h - self.min_h, -y], dtype=float)

    def place_world(self, pid, lift=0.0):
        p = self.places[pid]
        x, y = self.utm(p["lon"], p["lat"])
        return self.world(x, y, self.ground(x, y) + lift)


def sun_at(local: datetime):
    import forge3d as f3d
    u = local.astimezone(timezone.utc)
    s = f3d.sun_position_utc(LAT0, LON0, u.year, u.month, u.day, u.hour, u.minute, u.second)
    return float(s.azimuth), float(s.elevation)


LAKE_H = (1944.6, 1946.8)                  # Mono Lake's surface in the lidar, metres
LAKE_DEEP = np.array([0.10, 0.24, 0.33])    # what the lake looks like from the air, deep...
LAKE_SHALLOW = np.array([0.30, 0.46, 0.47])  # ...and over the shallow shelf by the shore


def lake_mask(dem_path, ref_tif, shape):
    """Mono Lake on a texture's grid: flat ground at the lake's surface height,
    joined to the biggest such area, softened at the shore. Also a 0-1
    'shallowness' that fades over the first ~150 m from the shore."""
    from rasterio.warp import reproject
    with rasterio.open(ref_tif) as r:
        b = r.bounds
    h, w = shape
    t = rasterio.transform.from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    z = np.full((h, w), np.nan, np.float32)
    with rasterio.open(dem_path) as d:
        reproject(rasterio.band(d, 1), z, dst_transform=t, dst_crs=CRS, resampling=rasterio.enums.Resampling.bilinear)
    flat = (z > LAKE_H[0]) & (z < LAKE_H[1])
    if not flat.any():
        return None, None
    lab, n = ndimage.label(flat)
    sizes = ndimage.sum(flat, lab, range(1, n + 1))
    px = (b.right - b.left) / w
    keep = np.flatnonzero(sizes * px * px > 2e5) + 1          # lake pieces over 0.2 km2
    if not keep.size:
        return None, None
    water = np.isin(lab, keep)
    water = ndimage.binary_opening(water, iterations=2)
    shore = ndimage.distance_transform_edt(water) * px
    alpha = np.clip(shore / 12.0, 0, 1)
    shallow = np.clip(1 - shore / 150.0, 0, 1) ** 1.5
    return alpha.astype(np.float32), shallow.astype(np.float32)


def grade_texture(src_tif: Path, out_png: Path, dem=None):
    """Aerial photo -> 8K texture, saturation x1.25 and a gentle S-curve (after
    the forge3d Bryce example: offsets the renderer's desaturating sky tint).
    With a DEM, Mono Lake is recoloured: NAIP catches the lake green with
    summer algae and glare; from the air on a clear morning it reads deep blue."""
    with rasterio.open(src_tif) as s:
        a = np.moveaxis(s.read(), 0, -1)
    im = Image.fromarray(a)
    w, h = im.size
    k = 8192 / max(w, h)
    im = im.resize((int(w * k), int(h * k)), Image.LANCZOS)
    a = np.asarray(im)
    alpha, shallow = lake_mask(dem, src_tif, a.shape[:2]) if dem is not None else (None, None)
    out = np.empty_like(a)
    wl = np.array([0.2126, 0.7152, 0.0722], np.float32)
    for r0 in range(0, a.shape[0], 512):
        rgb = a[r0:r0 + 512].astype(np.float32) / 255
        lum = (rgb @ wl)[..., None]
        rgb = np.clip(lum + (rgb - lum) * 1.25, 0, 1)
        rgb += 0.10 * (rgb - 0.5) * (1 - np.abs(2 * rgb - 1))
        if alpha is not None:
            al = alpha[r0:r0 + 512, :, None]
            sh = shallow[r0:r0 + 512, :, None]
            # keep the photo's ripples and shading as a faint texture on the new colour
            local = lum / (ndimage.uniform_filter(lum[..., 0], 41)[..., None] + 1e-3)
            lake = (LAKE_DEEP * (1 - sh) + LAKE_SHALLOW * sh) * np.clip(local, 0.85, 1.15)
            rgb = rgb * (1 - al) + lake * al
        out[r0:r0 + 512] = (np.clip(rgb * 0.92, 0, 1) * 255 + 0.5).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_png)


MAX_THETA = 85.0   # the viewer won't look flatter than 5 deg below horizontal


def effective_eye(eye, aim):
    """Where the viewer actually puts the camera: same target, radius and
    heading, with the polar angle clamped to 85 deg (forge3d viewer behaviour,
    measured with marker renders)."""
    off = eye - aim
    r = float(np.linalg.norm(off))
    theta = math.degrees(math.acos(off[1] / r))
    if theta <= MAX_THETA:
        return eye
    phi = math.atan2(off[2], off[0])
    t = math.radians(MAX_THETA)
    return aim + np.array([r * math.sin(t) * math.cos(phi), r * math.cos(t), r * math.sin(t) * math.sin(phi)])


def camera_cmd(eye, aim, fov):
    off = eye - aim
    r = float(np.linalg.norm(off))
    return {"cmd": "set_terrain_camera", "phi_deg": math.degrees(math.atan2(off[2], off[0])),
            "theta_deg": min(MAX_THETA, math.degrees(math.acos(off[1] / r))), "radius": r, "fov_deg": fov,
            "target": [float(v) for v in aim]}


def project(p, eye, aim, fov, size):
    eye = effective_eye(eye, aim)
    f = aim - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = p - eye
    z = float(d @ f)
    if z <= 1:
        return None
    th = math.tan(math.radians(fov) / 2)
    x = (d @ r) / (z * th * size[0] / size[1])
    y = (d @ u) / (z * th)
    return (x + 1) / 2 * size[0], (1 - y) / 2 * size[1], z


class Viewer:
    """One forge3d viewer holding one terrain layer (see layers.py)."""
    def __init__(self, scene: Scene, layer, size, fov):
        from forge3d.viewer import open_viewer_async
        self.size = size
        # the viewer puts y = 0 at its own DEM's minimum; cameras are in the scene's frame
        self.dy = layer.min_h - scene.min_h
        self.v = open_viewer_async(width=size[0], height=size[1], terrain_path=str(layer.dem),
                                   fov_deg=fov, timeout=600)
        self.v.load_overlay("naip", str(layer.texture), extent=layer.ext, z_order=0)
        self.v.send_ipc({"cmd": "set_terrain_pbr", "enabled": True, "exposure": 0.55, "shadow_map_res": 4096,
                         "height_ao": {"enabled": True, "strength": 0.8, "max_distance": 150.0},
                         "sun_visibility": {"enabled": True, "mode": "soft", "max_distance": 4000.0}})
        self.v.send_ipc({"cmd": "set_terrain", "ambient": 0.07, "zscale": 1.0})

    def sun(self, az, el):
        self.v.send_ipc({"cmd": "set_terrain_sun", "azimuth_deg": az, "elevation_deg": el, "intensity": 1.0})

    def shot(self, eye, aim, fov, path):
        d = np.array([0.0, self.dy, 0.0])
        self.v.send_ipc(camera_cmd(eye - d, aim - d, fov))
        self.v.snapshot(str(path), *self.size)

    def close(self):
        self.v.close()


class Stack:
    """Viewers for a far/mid/near layer stack, rendered and composited together."""
    def __init__(self, scene: Scene, layers, size, fov):
        self.layers = layers
        self.vs = []
        try:
            for L in layers:
                self.vs.append(Viewer(scene, L, size, fov))
        except Exception:
            self.close()
            raise

    def sun(self, az, el):
        for v in self.vs:
            v.sun(az, el)

    def shot(self, eye, aim, fov, path: Path):
        from layers import composite
        raws = []
        for L, v in zip(self.layers, self.vs):
            p = path.with_name(f"{path.stem}.{L.name}.png")
            v.shot(eye, aim, fov, p)
            raws.append(p)
        img = composite(raws)
        Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
        for p in raws:
            p.unlink()

    def close(self):
        for v in self.vs:
            v.close()


def stack_for(scene: Scene, cams, fov, size, sun, tag="", tiers=None):
    from layers import TIERS, layers_for
    return Stack(scene, layers_for(scene, cams, fov, size[0] / size[1], sun, scene.work, scene.source, tag,
                                   tiers or TIERS), size, fov)



# ── look ──────────────────────────────────────────────────────────────────────
@dataclass
class Sky:
    zenith: tuple
    horizon: tuple
    warm: tuple = (1.06, 1.0, 0.90)
    haze: float = 0.5


DAWN = Sky((0.16, 0.25, 0.46), (0.98, 0.79, 0.60))
WINTER = Sky((0.20, 0.32, 0.55), (0.95, 0.82, 0.70), (1.04, 1.0, 0.93))
DAY = Sky((0.25, 0.45, 0.75), (0.80, 0.86, 0.92), (1.0, 1.0, 0.98), 0.35)


def finish(png: Path, sky: Sky) -> Image.Image:
    """Paint the sky, add distance haze along the horizon, warm the light."""
    img = np.asarray(Image.open(png).convert("RGB")).astype(np.float32) / 255
    mask = sky_mask(img)
    rows = np.flatnonzero(~mask.all(axis=1))
    hz = int(rows[0]) if rows.size else img.shape[0] // 2
    H, W = img.shape[:2]
    yy = np.arange(H, dtype=np.float32)[:, None]
    t = np.clip(yy / max(hz, 1), 0, 1) ** 1.5
    zen, hor = np.array(sky.zenith), np.array(sky.horizon)
    skyimg = np.broadcast_to((zen * (1 - t) + hor * t)[:, None, :], (H, W, 3))
    ter = np.clip((img - 0.5) * 1.12 + 0.5, 0, 1) * np.array(sky.warm)
    k = (np.clip(1 - (yy - hz) / (H * 0.38), 0, 1) ** 2.4 * sky.haze)[..., None]
    ter = ter * (1 - k) + hor * k
    out = np.where(mask[..., None], skyimg, ter)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


def sky_mask(img: np.ndarray) -> np.ndarray:
    """The viewer's background is a smooth, pale gradient; terrain is textured
    and darker. Walk down each column until that stops being true."""
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H, W = lum.shape
    grad = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
    not_sky = (lum < 0.86) | (grad > 0.02)
    first = np.where(not_sky.any(axis=0), not_sky.argmax(axis=0), H)
    # smooth the silhouette a little, never letting it rise above the data
    return np.arange(H)[:, None] < first[None, :]


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def place_labels(items, size):
    """items: [(x, y, text, sub, alpha)] -> stem lengths that keep text boxes apart."""
    s = size[0] / 1920
    placed, out = [], []
    for x, y, text, sub, a in sorted(items, key=lambda t: -t[1]):
        w = (len(text) * 19 + 20) * s
        stem = 46 * s
        for _ in range(12):
            box = (x, y - stem - 34 * s, x + w, y - stem + (34 if sub else 6) * s)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                break
            stem += 30 * s
        placed.append(box)
        out.append((x, y, text, sub, a, stem / s))
    return out


def label(im: Image.Image, xy, text, sub=None, alpha=1.0, scale=1.0, stem_px=46):
    """A pin and a two-line label, after the page's typography."""
    if xy is None or alpha <= 0.01:
        return
    x, y = xy
    W, H = im.size
    if not (0 < x < W and 0 < y < H):
        return
    s = scale * W / 1920
    # Fade pins that are sliding off the frame instead of cutting them in half.
    edge = min(x, W - x) / (0.05 * W), (H - y) / (0.07 * H)
    alpha *= max(0.0, min(1.0, *edge))
    if alpha <= 0.01:
        return
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    stem = stem_px * s
    d.line([(x, y), (x, y - stem)], fill=(255, 244, 220, a), width=max(1, int(2 * s)))
    d.ellipse([x - 4 * s, y - 4 * s, x + 4 * s, y + 4 * s], fill=(255, 220, 160, a))
    f1 = font("cinzel-latin-600-normal.woff", int(30 * s))
    tx, ty = x + 10 * s, y - stem - 30 * s
    f2 = font("crimson-text-latin-400-italic.woff", int(22 * s))
    tw = max(d.textlength(text, font=f1), d.textlength(sub, font=f2) if sub else 0)
    if tx + tw > W - 12 * s:            # near the right edge: put the text left of the pin
        tx = x - 10 * s - tw
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        d.text((tx + dx * s, ty + dy * s), text, font=f1, fill=(20, 14, 8, int(a * 0.55)))
    d.text((tx, ty), text, font=f1, fill=(255, 246, 228, a))
    if sub:
        d.text((tx + 1, ty + 34 * s), sub, font=f2, fill=(20, 14, 8, int(a * 0.5)))
        d.text((tx, ty + 33 * s), sub, font=f2, fill=(250, 232, 200, a))
    im.paste(Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB"))


def caption(im: Image.Image, title, sub=None, alpha=1.0, where="bottom"):
    if alpha <= 0.01:
        return
    W, H = im.size
    s = W / 1920
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    f1 = font("cinzel-latin-400-normal.woff", int(44 * s))
    f2 = font("courier-prime-latin-400-normal.woff", int(20 * s))
    y = H - 150 * s if where == "bottom" else 70 * s
    grad = np.linspace(0, 0.55, int(220 * s)) if where == "bottom" else np.linspace(0.45, 0, int(200 * s))
    for i, g in enumerate(grad):
        yy = H - len(grad) + i if where == "bottom" else i
        d.line([(0, yy), (W, yy)], fill=(10, 7, 5, int(g * a)))
    d.text((70 * s, y), title, font=f1, fill=(240, 214, 160, a))
    if sub:
        d.text((72 * s, y + 60 * s), sub, font=f2, fill=(220, 200, 170, a))
    im.paste(Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB"))




MORNING = Sky((0.24, 0.42, 0.70), (0.88, 0.87, 0.84), (1.03, 1.0, 0.96), 0.24)


# ── flight ────────────────────────────────────────────────────────────────────
FPS = 30
FOV = 48.0
HOLD_S = 4.0
MIN_CLEARANCE = 90.0     # metres above the highest ground within LOOK_RADIUS of the eye
LOOK_RADIUS = 150.0

# (eye (lat, lon, alt m), aim (lat, lon, alt m), speed through this point m/s, lens fov deg, note)
# Each stretch takes its length / the average of its two speeds.
KEYS = [
    # 1. Hovering west of Tioga Pass, looking east across Tioga Lake to the head of Lee Vining Canyon.
    ((37.9020, -119.2760, 3520), (37.9360, -119.2050, 2900), 140, 50, "Tioga Pass"),
    ((37.9170, -119.2540, 3380), (37.9420, -119.1800, 2650), 400, 50, "over Tioga Lake"),
    # 2. Over Ellery Lake and down into Lee Vining Canyon, below its rims, Mono Lake at the far end.
    ((37.9370, -119.2330, 3150), (37.9430, -119.1950, 2500), 500, 50, "Ellery Lake"),
    ((37.9430, -119.2050, 2820), (37.9340, -119.1600, 2350), 520, 50, "Lee Vining Canyon"),
    ((37.9350, -119.1700, 2620), (37.9600, -119.1100, 2050), 560, 50, "lower canyon"),
    ((37.9450, -119.1350, 2400), (38.0000, -119.0600, 1950), 600, 50, "canyon mouth"),
    # 3. Out over the lake: Negit Island to the left, then along the north side of Paoha Island.
    ((37.9750, -119.0950, 2300), (38.0215, -119.0495, 2000), 620, 50, "the lake opens"),
    ((38.0100, -119.0750, 2200), (37.9984, -119.0363, 2000), 560, 46, "Negit Island"),
    ((38.0230, -119.0300, 2150), (37.9600, -119.0300, 1950), 500, 48, "Paoha Island"),
    # 4. A long turn south, down low to the South Tufa towers.
    ((37.9900, -118.9950, 2100), (37.9445, -119.0310, 1946), 420, 48, "turning south"),
    ((37.9550, -119.0150, 2020), (37.9420, -119.0330, 1946), 280, 40, "South Tufa"),
    ((37.9430, -119.0400, 2040), (37.9296, -119.0446, 2140), 320, 44, "to Panum Crater"),
    # 5. Over Panum Crater and south along the east side of the Mono Craters.
    ((37.9250, -119.0250, 2350), (37.8780, -119.0070, 2700), 420, 48, "Panum Crater"),
    ((37.9000, -118.9750, 2800), (37.8600, -119.0100, 2600), 450, 48, "Mono Craters"),
    # 6. Turning west, and settling on the craters with the whole Sierra escarpment behind them.
    ((37.8680, -118.9600, 3000), (37.9000, -119.2000, 3300), 200, 40, "turning west"),
    ((37.8580, -118.9550, 3050), (37.9050, -119.2150, 3400), 0, 36, "the escarpment"),
]

LABELS = [  # place, when it shows (between these two keyframe notes)
    ("tioga-pass", "Tioga Pass", "over Tioga Lake"),
    ("mount-dana", "Tioga Pass", "Ellery Lake"),
    ("tioga-lake", "Tioga Pass", "over Tioga Lake"),
    ("ellery-lake", "over Tioga Lake", "Lee Vining Canyon"),
    ("tioga-peak", "over Tioga Lake", "Lee Vining Canyon"),
    ("mono-lake", "lower canyon", "the lake opens"),
    ("lee-vining", "canyon mouth", "the lake opens"),
    ("negit-island", "the lake opens", "Paoha Island"),
    ("paoha-island", "Negit Island", "turning south"),
    ("south-tufa", "turning south", "to Panum Crater"),
    ("panum-crater", "South Tufa", "Panum Crater"),
    ("crater-mountain", "Panum Crater", "turning west"),
    ("mount-dana", "turning west", None),
    ("mount-gibbs", "turning west", None),
]


def catmull_rom(points: np.ndarray, samples: int = 400) -> np.ndarray:
    p = np.vstack([points[0], points, points[-1]])
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    seg = [0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
           for p0, p1, p2, p3 in zip(p[:-3], p[1:-2], p[2:-1], p[3:])]
    return np.vstack(seg + [points[-1:]])


@dataclass
class Flight:
    t: np.ndarray
    eye: np.ndarray        # (n, 3) UTM x, y, altitude
    aim: np.ndarray
    fov: np.ndarray
    clearance: np.ndarray
    speed: np.ndarray
    lifted: float
    key_t: np.ndarray


def plan(sc: Scene | None, fps: int = FPS) -> Flight:
    from scipy.interpolate import PchipInterpolator
    utm = lambda lat, lon: transform("EPSG:4326", CRS, [lon], [lat])
    ek = np.array([[*[v[0] for v in utm(k[0][0], k[0][1])], k[0][2]] for k in KEYS])
    ak = np.array([[*[v[0] for v in utm(k[1][0], k[1][1])], k[1][2]] for k in KEYS])
    samples = 400
    dense = catmull_rom(ek, samples)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    s_k = arc[np.arange(len(KEYS)) * samples]
    v = np.array([k[2] for k in KEYS], float)
    times = np.concatenate([[0.0], np.cumsum(np.diff(s_k) / np.maximum((v[:-1] + v[1:]) / 2, 20.0))])
    times = np.append(times, times[-1] + HOLD_S)
    s_k = np.append(s_k, s_k[-1])
    ak = np.vstack([ak, ak[-1]])
    fov_k = np.array([k[3] for k in KEYS] + [KEYS[-1][3]], float)
    s_of_t = PchipInterpolator(times, s_k)
    n = int(round(times[-1] * fps)) + 1
    t = np.arange(n) / fps
    s = s_of_t(t)
    eye = np.column_stack([np.interp(s, arc, dense[:, j]) for j in range(3)])
    aim = np.column_stack([PchipInterpolator(times, ak[:, j])(t) for j in range(3)])
    fov = PchipInterpolator(times, fov_k)(t)
    speed = np.abs(s_of_t(t, 1))
    lifted, clear = 0.0, np.full(n, np.nan)
    if sc is not None:
        top = np.array([sc.ground_max(x, y, LOOK_RADIUS) for x, y in eye[:, :2]])
        need = np.maximum(0.0, top + MIN_CLEARANCE - eye[:, 2])
        if need.max() > 0:
            lift = ndimage.maximum_filter1d(need, size=2 * fps + 1)
            lift = np.maximum(ndimage.gaussian_filter1d(lift, fps * 0.8), need)
            eye[:, 2] += lift
            lifted = float(lift.max())
        clear = eye[:, 2] - np.array([sc.ground(x, y) for x, y in eye[:, :2]])
    return Flight(t, eye, aim, fov, clear, speed, lifted, times[:-1])


def key_time(f: Flight, note):
    if note is None:
        return f.t[-1] + 1
    return float(f.key_t[[k[4] for k in KEYS].index(note)])


def report(f: Flight):
    print(f"{len(f.t)} frames, {f.t[-1]:.0f} s at {FPS} fps")
    for t0, k in zip(f.key_t, KEYS):
        i = min(int(round(t0 * FPS)), len(f.t) - 1)
        c = f"{f.clearance[i]:5.0f} m up" if np.isfinite(f.clearance[i]) else ""
        print(f"  {t0:5.1f} s  {k[4]:<20} {f.speed[i]:4.0f} m/s  lens {f.fov[i]:3.0f}  {c}")
    if np.isfinite(f.clearance).any():
        i = int(np.nanargmin(f.clearance))
        print(f"  lowest {f.clearance[i]:.0f} m above the ground at {f.t[i]:.1f} s; raised by up to {f.lifted:.0f} m")


def path_map(sc: Scene, f: Flight, dst: Path):
    """Hillshade (prep DEM) with the flight: track coloured by time, aim lines every 2 s."""
    step = 4
    z = sc.dem[::step, ::step].astype(float)
    gy, gx = np.gradient(z, sc.res * step)
    az, el = np.radians(315), np.radians(40)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    shade = np.sin(el) * np.cos(slope) + np.cos(el) * np.sin(slope) * np.cos(az - aspect)
    zn = (z - z.min()) / (z.max() - z.min())
    base = np.clip(0.25 + 0.6 * shade, 0, 1)[..., None] * (0.55 + 0.45 * zn[..., None]) * np.array([235, 230, 220])
    img = Image.fromarray(base.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    inv = ~sc.T
    px = lambda x, y: tuple(v / step for v in inv * (x, y))
    n = len(f.t)
    for i in range(0, n, FPS * 2):
        d.line([px(*f.eye[i, :2]), px(*f.aim[i, :2])], fill=(255, 255, 255), width=1)
    for i in range(n - 1):
        u = i / max(n - 1, 1)
        d.line([px(*f.eye[i, :2]), px(*f.eye[i + 1, :2])], fill=(int(255 * u), int(80 + 100 * (1 - u)), int(255 * (1 - u))), width=3)
    for t0, k in zip(f.key_t, KEYS):
        i = min(int(round(t0 * FPS)), n - 1)
        x, y = px(*f.eye[i, :2])
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 0))
        d.text((x + 6, y - 6), f"{t0:.0f}s {k[4]}", fill=(255, 255, 0))
    for p in sc.places.values():
        x, y = px(*sc.utm(p["lon"], p["lat"]))
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 90, 180))
        d.text((x + 5, y + 3), p["name"], fill=(255, 140, 200))
    img.save(dst)
    print(f"wrote {dst}")


def frame_labels(sc: Scene, f: Flight, i: int, size):
    t = f.t[i]
    eye, aim = sc.world(*f.eye[i]), sc.world(*f.aim[i])
    items = []
    for pid, a, b in LABELS:
        ta, tb = key_time(f, a), key_time(f, b)
        fade = min(1.0, (t - ta) / 1.0, (tb - t) / 1.0)
        if fade <= 0:
            continue
        p = project(sc.place_world(pid), eye, aim, float(f.fov[i]), size)
        if p and p[2] < 40000:
            pl = sc.places[pid]
            items.append((p[0], p[1], pl["name"], pl.get("sub"), fade))
    return items


def draw_frame(sc: Scene, f: Flight, i: int, raw: Path, size) -> Image.Image:
    im = finish(raw, MORNING)
    for x, y, text, sub, a, stem in place_labels(frame_labels(sc, f, i, size), size):
        label(im, (x, y), text, sub, a, stem_px=stem)
    t = f.t[i]
    ta = min(1.0, max(0.0, min((t - 0.3) / 1.0, (7.0 - t) / 1.0)))
    caption(im, "Tioga Pass to Mono Lake", "a September morning · USGS 3DEP lidar · NAIP aerial photography · forge3d", alpha=ta)
    tb = min(1.0, max(0.0, (t - (f.t[-1] - 4.5)) / 1.0))
    caption(im, "The Mono Basin", "Mono Craters, and the Sierra Nevada's eastern escarpment", alpha=tb)
    return im


def flyover(sc: Scene, out: Path, chunk: int, chunks: int, size=(1920, 1080)):
    f = plan(sc)
    n = len(f.t)
    lo, hi = chunk * n // chunks, (chunk + 1) * n // chunks
    out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(lo, hi) if not (out / f"frame_{i:05d}.jpg").exists()]
    if not todo:
        print(f"chunk {chunk}: all {hi - lo} frames already rendered", flush=True)
        return
    t0 = time.time()
    sun = sun_at(WHEN)
    cams = [(sc.world(*f.eye[i]), sc.world(*f.aim[i])) for i in list(range(lo, hi, 4)) + [hi - 1]]
    fov = float(f.fov[lo:hi].max())
    v = stack_for(sc, cams, fov, size, sun, tag=f"-c{chunk:02d}")
    print(f"chunk {chunk}: frames {lo}-{hi - 1}, layers ready {time.time() - t0:.0f}s", flush=True)
    v.sun(*sun)
    try:
        cap = int(os.environ.get("MAX_FRAMES", "0") or 0)
        for i in (todo[:cap] if cap else todo):
            raw = out / f"raw_{i:05d}.png"
            v.shot(sc.world(*f.eye[i]), sc.world(*f.aim[i]), float(f.fov[i]), raw)
            draw_frame(sc, f, i, raw, size).save(out / f"frame_{i:05d}.jpg", quality=93)
            raw.unlink()
            if (i - lo) % 10 == 0:
                print(f"frame {i} ({i - lo + 1}/{hi - lo}) {time.time() - t0:.0f}s", flush=True)
    finally:
        v.close()


def stills(sc: Scene, out: Path, seconds, size=(1920, 1080)):
    f = plan(sc)
    sun = sun_at(WHEN)
    print(f"sun {WHEN:%Y-%m-%d %H:%M %Z}: azimuth {sun[0]:.0f}, elevation {sun[1]:.0f}", flush=True)
    for s in seconds:
        i = min(int(round(s * FPS)), len(f.t) - 1)
        eye, aim = sc.world(*f.eye[i]), sc.world(*f.aim[i])
        t = time.time()
        v = stack_for(sc, [(eye, aim)], float(f.fov[i]), size, sun, tag=f"-s{i:05d}")
        try:
            v.sun(*sun)
            raw = out / f"raw_{i:05d}.png"
            v.shot(eye, aim, float(f.fov[i]), raw)
        finally:
            v.close()
        draw_frame(sc, f, i, raw, size).save(out / f"still_{s:05.1f}s.jpg", quality=90)
        raw.unlink()
        print(f"still {s} s: {time.time() - t:.0f}s", flush=True)


def encode(frames: Path, out: Path):
    ff = shutil.which("ffmpeg")
    out.parent.mkdir(parents=True, exist_ok=True)
    common = ["-framerate", str(FPS), "-i", str(frames / "frame_%05d.jpg"), "-pix_fmt", "yuv420p",
              "-movflags", "+faststart", "-c:v", "libx264", "-preset", "slow"]
    subprocess.run([ff, "-y", "-loglevel", "error", *common, "-crf", "18", str(out.with_name(out.stem + "-1080.mp4"))], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", *common, "-crf", "25", "-vf", "scale=1280:-2", str(out)], check=True)
    frames_l = sorted(frames.glob("frame_*.jpg"))
    poster = frames_l[int(len(frames_l) * 0.93)]
    Image.open(poster).resize((1280, 720), Image.LANCZOS).save(out.with_name(out.stem + "-poster.jpg"), quality=86)
    print(f"wrote {out} and {out.stem}-1080.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["path", "stills", "flyover", "encode"])
    ap.add_argument("--prep", default="prep")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--work", default="work")
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--all", action="store_true", help="flyover: every chunk in turn on this machine")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--seconds", default="0,12,19,26,34,42,50,60,72,80,90,100,112,128")
    ap.add_argument("--source", choices=["usgs", "prep"], default="usgs")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    size = (a.width, a.width * 9 // 16)
    if a.mode == "encode":
        return encode(Path(a.frames), out / "mono-flyover.mp4")
    sc = Scene(Path(a.prep), work, a.source)
    if a.mode == "path":
        f = plan(sc)
        report(f)
        path_map(sc, f, out / "flight_map.png")
    elif a.mode == "stills":
        stills(sc, out, [float(s) for s in a.seconds.split(",")], size)
    elif a.mode == "flyover":
        for c in (range(a.chunks) if a.all else [a.chunk]):
            flyover(sc, Path(a.frames), c, a.chunks, size)


if __name__ == "__main__":
    main()
