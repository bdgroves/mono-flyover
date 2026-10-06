"""
Nested terrain layers, so the renders can use the full 1 m lidar.

forge3d's viewer meshes at most 2048 vertices across a terrain and point-samples
anything bigger down to fit (tools/README note; measured with synthetic DEMs).
Over the whole 36 km render extent that is ~18 m a vertex, whatever the source
resolution. So each frame is drawn in layers, each one under the cap:

  far   the whole extent, averaged down to ~11 m       (the horizon)
  mid   what the camera sees within MID_M,  a few m     (the canyon ahead)
  near  what the camera sees within NEAR_M, 1-2 m       (the foreground)

Every layer is rendered with the same camera and Sun; the finer layer wins
wherever it has terrain, and the coarser one shows through beyond its edge.
Each fine window reaches toward the Sun far enough to keep the long dawn
shadows that fall into it.

Beneath them all sits a horizon layer (~170 km across at ~85 m, Sentinel-2
colour) so the far skyline is real terrain, not the edge of the data.

Windows are cut from USGS 3DEP (1 m lidar here: CA_SierraNevada_B22 and
CA_YosemiteNP_2019) and NAIP at its native 0.6 m, fetched for each window
(source="usgs"). source="prep" cuts them from the 8 m prep data instead, for
testing where the data services can't be reached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

MAX_VERTS = 2048          # the viewer's cap on vertices across
NEAR_M, MID_M = 1300.0, 4500.0
LIDAR_M = 1.0             # finest DEM step worth asking for from 3DEP
DENSE_M = 0.5             # ... and where the 0.5 m CONMGaps tiles cover (prep_lidar.py)
FEATHER_M = 40.0          # blend lidar tiles into the 3DEP surface over this distance
NAIP_M = 0.6              # NAIP's native ground sample
TEX_MAX = 8192            # overlay texture side


@dataclass
class Layer:
    name: str
    dem: Path
    texture: Path
    ext: tuple            # overlay extent as fractions of the DEM
    min_h: float          # the viewer puts y = 0 at the DEM minimum
    step: float
    bounds: tuple         # UTM x0, y0, x1, y1


def snap(bounds, step):
    x0, y0, x1, y1 = bounds
    return (math.floor(x0 / step) * step, math.floor(y0 / step) * step,
            math.ceil(x1 / step) * step, math.ceil(y1 / step) * step)


def step_for(bounds, finest):
    x0, y0, x1, y1 = bounds
    s = max(x1 - x0, y1 - y0) / (MAX_VERTS - 2)
    return max(finest, math.ceil(s * 4) / 4)          # quarter-metre steps


def finest(scene, source):
    if getattr(scene, "lidar", None):
        return DENSE_M
    return LIDAR_M if source == "usgs" else 8.0


def patch_lidar(scene, path, label=""):
    """Lay the 0.5 m ground tiles over a DEM in place, feathered at their edges."""
    from scipy.ndimage import distance_transform_edt
    tiles = getattr(scene, "lidar", None)
    if not tiles:
        return
    with rasterio.open(path) as d:
        a, T, prof, b, step = d.read(1), d.transform, d.profile, d.bounds, d.res[0]
    acc = np.full(a.shape, np.nan, dtype="float32")
    how = Resampling.average if step >= 1.0 else Resampling.bilinear
    for t in tiles:
        x0, y0, x1, y1 = t["bounds"]
        if x1 <= b.left or x0 >= b.right or y1 <= b.bottom or y0 >= b.top:
            continue
        tmp = np.full(a.shape, np.nan, dtype="float32")
        with rasterio.open(t["path"]) as s:
            reproject(rasterio.band(s, 1), tmp, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                      dst_transform=T, dst_crs=prof["crs"], dst_nodata=np.nan, resampling=how)
        put = np.isnan(acc) & np.isfinite(tmp)
        acc[put] = tmp[put]
    have = np.isfinite(acc)
    if not have.any():
        return
    both = have & np.isfinite(a)
    off = float(np.median(acc[both] - a[both])) if both.any() else float("nan")
    w = np.clip(distance_transform_edt(have) * step / FEATHER_M, 0, 1).astype("float32")
    w[np.isnan(a) & have] = 1.0
    out = np.where(have, np.nan_to_num(a) * (1 - w) + np.nan_to_num(acc) * w, a).astype("float32")
    prof.update(dtype="float32", nodata=None)
    with rasterio.open(path, "w", **prof) as d:
        d.write(out, 1)
    print(f"  {label or path.name}: 0.5 m lidar over {have.mean():.0%} of it "
          f"(median {off:+.2f} m from 3DEP)", flush=True)


def write_dem(path, a, x0, y1, step, crs):
    prof = dict(driver="GTiff", width=a.shape[1], height=a.shape[0], count=1, dtype="float32",
                crs=crs, transform=from_origin(x0, y1, step, step), compress="deflate",
                predictor=3, tiled=True)
    with rasterio.open(path, "w", **prof) as d:
        d.write(a.astype("float32"), 1)


def resample(src_path, bounds, step, out, crs, bands=1, how=Resampling.average):
    x0, y0, x1, y1 = bounds
    w, h = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    t = from_origin(x0, y1, step, step)
    with rasterio.open(src_path) as s:
        dst = np.zeros((bands, h, w), dtype="float32" if bands == 1 else "uint8")
        for b in range(bands):
            reproject(rasterio.band(s, b + 1), dst[b], src_transform=s.transform, src_crs=s.crs,
                      dst_transform=t, dst_crs=crs, resampling=how)
    return dst, t


# ── layers ────────────────────────────────────────────────────────────────────
def far_layer(scene, work: Path) -> Layer:
    """The whole extent, averaged (not point-sampled) to fit the cap."""
    B = scene.B
    step = step_for((B.left, B.bottom, B.right, B.top), 8.0)
    bounds = snap((B.left, B.bottom, B.right, B.top), step)
    path = work / f"far_dem_{step:g}m{'_lidar' if getattr(scene, 'lidar', None) else ''}.tif"
    if not path.exists():
        a, _ = resample(scene.dem_src, bounds, step, None, scene.crs)
        write_dem(path, a[0], bounds[0], bounds[3], step, scene.crs)
    with rasterio.open(path) as s:
        mn = float(np.nanmin(s.read(1)))
    # the existing 8K texture covers the prep extent; express it in this grid
    nb = scene.naip_bounds
    W, H = bounds[2] - bounds[0], bounds[3] - bounds[1]
    ext = ((nb.left - bounds[0]) / W, (bounds[3] - nb.top) / H,
           (nb.right - bounds[0]) / W, (bounds[3] - nb.bottom) / H)
    return Layer("far", path, scene.texture, ext, mn, step, bounds)


def horizon_layer(scene, work: Path) -> Layer:
    """The distant skyline: ~170 km of terrain at ~85 m, Sentinel-2 colour."""
    from render import grade_texture
    with rasterio.open(scene.horizon_src) as s:
        B = s.bounds
    step = step_for((B.left, B.bottom, B.right, B.top), 60.0)
    bounds = snap((B.left, B.bottom, B.right, B.top), step)
    path = work / f"horizon_dem_{step:g}m.tif"
    if not path.exists():
        a, _ = resample(scene.horizon_src, bounds, step, None, scene.crs)
        write_dem(path, a[0], bounds[0], bounds[3], step, scene.crs)
    with rasterio.open(path) as s:
        mn = float(np.nanmin(s.read(1)))
    tex = work / "horizon_tex.png"
    if not tex.exists():
        grade_texture(scene.horizon_tex_src, tex, scene.horizon_src)
    with rasterio.open(scene.horizon_tex_src) as s:
        nb = s.bounds
    W, H = bounds[2] - bounds[0], bounds[3] - bounds[1]
    ext = ((nb.left - bounds[0]) / W, (bounds[3] - nb.top) / H, (nb.right - bounds[0]) / W, (bounds[3] - nb.bottom) / H)
    return Layer("horizon", path, tex, ext, mn, step, bounds)


def window_layer(scene, name: str, bounds, work: Path, source: str) -> Layer:
    from render import grade_texture
    step = step_for(bounds, finest(scene, source))
    bounds = snap(bounds, step)
    key = f"{name}_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}_{step:g}"
    dem = work / f"{key}_dem.tif"
    tif = work / f"{key}_naip.tif"
    tex = work / f"{key}_tex.png"
    tstep = max(NAIP_M if source == "usgs" else 5.0, max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / TEX_MAX)
    if not dem.exists():
        fetched = False
        if source == "usgs":
            from prep_data import fetch_dem_utm
            # smaller requests on failure: the ImageServer 502s on some windows
            for tile in (2000, 1000, 500):
                try:
                    fetch_dem_utm(bounds, step, dem, tile=tile)
                    fetched = True
                    break
                except RuntimeError as e:
                    print(f"  {name}: 3DEP export failed at {tile}-px requests ({e})", flush=True)
            if not fetched:
                print(f"  {name}: falling back to the 5 m prep DEM (the 0.5 m lidar patches over it)", flush=True)
        if not fetched:
            a, _ = resample(scene.dem_src, bounds, step, None, scene.crs, how=Resampling.bilinear)
            write_dem(dem, a[0], bounds[0], bounds[3], step, scene.crs)
        patch_lidar(scene, dem, name)
    if not tex.exists():
        if source == "usgs":
            fetch_naip_utm(bounds, tstep, tif, scene.crs)
        else:
            a, t = resample(scene.naip_src, bounds, tstep, None, scene.crs, bands=3, how=Resampling.bilinear)
            prof = dict(driver="GTiff", width=a.shape[2], height=a.shape[1], count=3, dtype="uint8",
                        crs=scene.crs, transform=t)
            with rasterio.open(tif, "w", **prof) as d:
                d.write(a)
        grade_texture(tif, tex, dem)
    with rasterio.open(dem) as s:
        a = s.read(1)
        mn = float(np.nanmin(a))
        if np.isnan(a).any():
            print(f"  {name}: {np.isnan(a).mean():.3%} of the window has no data", flush=True)
    print(f"  {name} layer: {bounds[2] - bounds[0]:.0f} x {bounds[3] - bounds[1]:.0f} m at {step:g} m"
          f" (texture {tstep:.2f} m)", flush=True)
    return Layer(name, dem, tex, (0.0, 0.0, 1.0, 1.0), mn, step, bounds)


def fetch_naip_utm(bounds, step, path, crs):
    """NAIP at `step` over a UTM window, reading only the blocks it needs."""
    import json
    from rasterio.vrt import WarpedVRT
    from prep_data import http_get
    pc = "https://planetarycomputer.microsoft.com/api"
    ll = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    import time

    def get(url, data=None):
        for attempt in range(6):
            try:
                code, body = http_get(url, data=data)
            except OSError as e:
                code, body = 0, str(e).encode()
            if code == 200:
                return body
            print(f"  NAIP retry {attempt + 1}: HTTP {code} {body[:80]!r}", flush=True)
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"NAIP: HTTP {code} {body[:200]!r}")

    body = get(f"{pc}/stac/v1/search", json.dumps({"collections": ["naip"], "bbox": list(ll), "limit": 200}).encode())
    feats = json.loads(body)["features"]
    year = lambda f: int(f["properties"].get("naip:year") or f["properties"]["datetime"][:4])
    best = max(year(f) for f in feats)
    feats = [f for f in feats if year(f) == best]
    body = get(f"{pc}/sas/v1/token/naip")
    token = json.loads(body)["token"]
    hrefs = [f["assets"]["image"]["href"] + "?" + token for f in feats]
    x0, y0, x1, y1 = bounds
    w, h = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    t = from_origin(x0, y1, step, step)
    mosaic = np.zeros((3, h, w), dtype="uint8")
    filled = np.zeros((h, w), dtype=bool)
    for href in hrefs:
        for attempt in range(4):
            try:
                with rasterio.open(href) as src, \
                        WarpedVRT(src, crs=crs, transform=t, width=w, height=h,
                                  resampling=Resampling.average, nodata=0) as vrt:
                    part = vrt.read(indexes=[1, 2, 3])
                break
            except rasterio.errors.RasterioIOError as e:
                print(f"  NAIP read retry {attempt + 1}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError("NAIP read failed")
        have = (part.sum(axis=0) > 0) & ~filled
        mosaic[:, have] = part[:, have]
        filled |= have
    print(f"  NAIP {best}: {len(hrefs)} quarter-quads, {filled.mean():.1%} filled", flush=True)
    prof = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=crs, transform=t)
    with rasterio.open(path, "w", **prof) as d:
        d.write(mosaic)


# ── what the camera sees ──────────────────────────────────────────────────────
def footprint(scene, cams, fov, aspect, dmax, nx=41, ny=23):
    """Ground points (UTM x, y) visible within `dmax` of the camera, for each
    (eye, aim) in `cams` (viewer world coordinates)."""
    from render import effective_eye
    dem, T = scene.dem, scene.T
    inv = ~T
    th = math.tan(math.radians(fov) / 2)
    sx, sy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    d = np.arange(10.0, dmax + 1, 10.0)
    pts = []
    for eye, aim in cams:
        eye = effective_eye(np.asarray(eye, float), np.asarray(aim, float))
        f = aim - eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        dirs = f + (sx.ravel()[:, None] * th * aspect) * r + (sy.ravel()[:, None] * th) * u
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        P = eye[None, None, :] + dirs[:, None, :] * d[None, :, None]       # rays x steps x 3
        X, N, Hh = P[..., 0], -P[..., 2], P[..., 1] + scene.min_h
        c, rr = inv * (X, N)
        rr, c = np.floor(rr).astype(int), np.floor(c).astype(int)
        ok = (rr >= 0) & (rr < dem.shape[0]) & (c >= 0) & (c < dem.shape[1])
        g = np.full(X.shape, np.inf)
        g[ok] = dem[rr[ok], c[ok]]
        hit = Hh <= g
        first = hit.argmax(axis=1)
        anyhit = hit.any(axis=1) & ok[np.arange(len(first)), first]
        idx = np.flatnonzero(anyhit)
        pts.append(np.c_[X[idx, first[idx]], N[idx, first[idx]]])
        pts.append(np.array([[eye[0], -eye[2]]]))           # and under the camera
    return np.vstack(pts)


def shadow_reach(scene, pts, sun_az, sun_el, cap=3000.0):
    """How far toward the Sun the ground that can shade `pts` reaches: the
    farthest point along the Sun's bearing that stands above a line climbing
    from each point at the Sun's elevation. Measured on the 5 m DEM."""
    dem, inv = scene.dem, ~scene.T
    if len(pts) > 600:
        pts = pts[np.random.default_rng(0).choice(len(pts), 600, replace=False)]
    ux, uy = math.sin(math.radians(sun_az)), math.cos(math.radians(sun_az))
    d = np.arange(0.0, cap + 1, 20.0)
    X = pts[:, 0:1] + ux * d[None, :]
    Y = pts[:, 1:2] + uy * d[None, :]
    c, r = inv * (X, Y)
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    ok = (r >= 0) & (r < dem.shape[0]) & (c >= 0) & (c < dem.shape[1])
    g = np.full(X.shape, -np.inf)
    g[ok] = dem[r[ok], c[ok]]
    g = np.nan_to_num(g, nan=-np.inf)
    h0 = g[:, :1]
    shades = (g - h0) > d[None, :] * math.tan(math.radians(max(sun_el, 0.5)))
    shades[:, 0] = False
    far = np.where(shades.any(axis=0))[0]
    return float(d[far.max()]) + 100.0 if far.size else 100.0


def window(scene, pts, sun_az, sun_el, margin=300.0, shadow_max=3000.0):
    """Bounding box of `pts`, padded, and stretched toward the Sun as far as the
    terrain that can actually shade it (see shadow_reach), clipped to the extent."""
    x0, y0 = pts.min(axis=0) - margin
    x1, y1 = pts.max(axis=0) + margin
    reach = min(shadow_max, shadow_reach(scene, pts, sun_az, sun_el, shadow_max))
    dx, dy = reach * math.sin(math.radians(sun_az)), reach * math.cos(math.radians(sun_az))
    x0, x1 = min(x0, x0 + dx), max(x1, x1 + dx)
    y0, y1 = min(y0, y0 + dy), max(y1, y1 + dy)
    B = scene.B
    return (max(x0, B.left), max(y0, B.bottom), min(x1, B.right), min(y1, B.top))


TIERS = (("mid", MID_M), ("near", NEAR_M))     # plus horizon and far beneath


def layers_for(scene, cams, fov, aspect, sun, work, source, tag="", tiers=TIERS):
    """far + finer window layers (by default mid and near) for a set of cameras
    (one shot or one chunk of the flight)."""
    out = [scene.horizon, scene.far]
    for name, dmax in tiers:
        pts = footprint(scene, cams, fov, aspect, dmax)
        b = window(scene, pts, *sun, margin=min(300.0, 0.15 * dmax))
        step = step_for(b, finest(scene, source))
        if step >= out[-1].step * 0.8:      # no finer than the layer below: skip
            continue
        out.append(window_layer(scene, name + tag, b, work, source))
    return out


def composite(raws):
    """Coarse to fine: each finer render replaces the one below wherever it has terrain."""
    from PIL import Image
    from render import sky_mask
    base = np.asarray(Image.open(raws[0]).convert("RGB")).astype(np.float32) / 255
    for p in raws[1:]:
        img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
        ter = ~sky_mask(img)
        base = np.where(ter[..., None], img, base)
    return base
