"""
Fetch the base terrain and imagery for the Mono Basin flyover. Runs on GitHub
Actions (prep-data.yml), which can reach the data services; also runs locally.

Writes to prep/:
  mono_dem_8m.tif         USGS 3DEP best-available elevation (1 m lidar here,
                          averaged to 8 m), UTM 11N, Tioga Pass to the east
                          shore of Mono Lake, the lake's north shore to the
                          south end of the Mono Craters
  mono_naip_5m.tif        USDA NAIP aerial photography mosaic, 5 m, same extent
  horizon_dem_60m.tif     3DEP at 60 m, ~80 km each way: the distant skyline
  horizon_s2_60m.tif      a clear late-September Sentinel-2 true-colour scene
                          for the horizon layer's colour, 60 m
  sources.json            what was fetched, from where, when

The fine terrain the camera flies close to (1 m lidar, 0.6 m NAIP) is fetched
per window at render time (layers.py).
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

OUT = Path("prep")
EPSG = 26911
UTM = f"EPSG:{EPSG}"   # NAD83 / UTM 11N
IMAGESERVER = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"

RENDER_LL = (-119.34, 37.80, -118.93, 38.08)
HORIZON_LL = (-120.15, 37.30, -118.15, 38.60)


def http_get(url, params=None, data=None, timeout=300):
    """(status, body bytes), standard library only."""
    import urllib.error
    import urllib.parse
    import urllib.request
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "mono-flyover (brooksgroves.com)",
                                 **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def utm_box(ll, step):
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", UTM, *ll, densify_pts=21)
    x0, y0 = math.floor(x0 / step) * step, math.floor(y0 / step) * step
    x1, y1 = math.ceil(x1 / step) * step, math.ceil(y1 / step) * step
    return x0, y0, x1, y1


def fetch_dem_utm(bounds, step, path, tile=2000):
    """3DEP best-available elevation over a UTM 11N box at `step` metres."""
    name = Path(path).name
    x0, y0, x1, y1 = bounds
    w, h = int(round((x1 - x0) / step)), int(round((y1 - y0) / step))
    out = np.full((h, w), np.nan, dtype="float32")
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            tw, th = min(tile, w - c0), min(tile, h - r0)
            bx0 = x0 + c0 * step
            by1 = y1 - r0 * step
            params = dict(bbox=f"{bx0},{by1 - th * step},{bx0 + tw * step},{by1}", bboxSR=EPSG, imageSR=EPSG,
                          size=f"{tw},{th}", format="tiff", pixelType="F32", noData=-9999,
                          interpolation="RSP_BilinearInterpolation", f="image")
            for attempt in range(6):
                try:
                    code, body = http_get(IMAGESERVER, params)
                except OSError as e:
                    code, body = 0, str(e).encode()
                if code == 200 and body[:2] in (b"II", b"MM"):
                    break
                print(f"  retry {name} tile {r0},{c0}: HTTP {code} {body[:80]!r}", flush=True)
                time.sleep(4 * (attempt + 1))
            else:
                raise RuntimeError(f"3DEP tile failed {r0},{c0}")
            with rasterio.MemoryFile(body) as mf, mf.open() as src:
                a = src.read(1).astype("float32")
                a[a < -1000] = np.nan
                out[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a[:th, :tw]
    if np.isnan(out).any():
        from scipy import ndimage
        hole = np.isnan(out)
        idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
        out = out[idx[0], idx[1]]
        print(f"  {name}: filled {hole.mean():.4%} empty cells from neighbours", flush=True)
    prof = dict(driver="GTiff", width=w, height=h, count=1, dtype="float32", crs=UTM,
                transform=from_origin(x0, y1, step, step), compress="deflate", predictor=3, tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
    print(f"wrote {path} {w}x{h} at {step:g} m, {np.nanmin(out):.0f}-{np.nanmax(out):.0f} m", flush=True)
    return path


def fetch_naip(ll, step, name):
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["naip"], bbox=ll).items())
    yr = lambda i: int(i.properties.get("naip:year") or i.datetime.year)
    years = sorted({yr(i) for i in items})
    best = max(years)
    items = [i for i in items if yr(i) == best]
    print(f"NAIP years {years}; using {best}: {len(items)} quarter-quads", flush=True)
    x0, y0, x1, y1 = utm_box(ll, step)
    w, h = int((x1 - x0) / step), int((y1 - y0) / step)
    t = from_origin(x0, y1, step, step)
    mosaic = np.zeros((3, h, w), dtype="uint8")
    filled = np.zeros((h, w), dtype=bool)
    from rasterio.vrt import WarpedVRT
    for it in items:
        for attempt in range(4):
            try:
                with rasterio.open(it.assets["image"].href) as src, \
                        WarpedVRT(src, crs=UTM, transform=t, width=w, height=h,
                                  resampling=Resampling.average, nodata=0) as v:
                    part = v.read(indexes=[1, 2, 3])
                break
            except rasterio.errors.RasterioIOError as e:
                print(f"  retry {it.id}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"NAIP read failed: {it.id}")
        have = (part.sum(axis=0) > 0) & ~filled
        mosaic[:, have] = part[:, have]
        filled |= have
    print(f"  NAIP {filled.mean():.2%} filled", flush=True)
    prof = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=UTM, transform=t,
                compress="jpeg", jpeg_quality=90, photometric="ycbcr", tiled=True)
    with rasterio.open(OUT / name, "w", **prof) as dst:
        dst.write(mosaic)
    print(f"wrote {name} {w}x{h}", flush=True)
    return best, [i.id for i in items]


def fetch_s2(ll, step, name, start="2025-09-15", end="2025-10-10"):
    """The clearest Sentinel-2 L2A true-colour (TCI) scenes over the horizon
    extent in late September, mosaicked at `step` metres."""
    import planetary_computer
    import pystac_client
    from rasterio.vrt import WarpedVRT
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=ll, datetime=f"{start}/{end}",
                            query={"eo:cloud_cover": {"lt": 10}}).items())
    items.sort(key=lambda i: i.properties["eo:cloud_cover"])
    x0, y0, x1, y1 = utm_box(ll, step)
    w, h = int((x1 - x0) / step), int((y1 - y0) / step)
    t = from_origin(x0, y1, step, step)
    mosaic = np.zeros((3, h, w), dtype="uint8")
    filled = np.zeros((h, w), dtype=bool)
    used = []
    for it in items:
        if filled.all():
            break
        with rasterio.open(it.assets["visual"].href) as src, \
                WarpedVRT(src, crs=UTM, transform=t, width=w, height=h,
                          resampling=Resampling.average, nodata=0) as v:
            part = v.read(indexes=[1, 2, 3])
        have = (part.min(axis=0) > 0) & ~filled
        if have.any():
            mosaic[:, have] = part[:, have]
            filled |= have
            used.append(it.id)
    print(f"  Sentinel-2: {len(used)} scenes, {filled.mean():.2%} filled", flush=True)
    prof = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=UTM, transform=t,
                compress="deflate", tiled=True)
    with rasterio.open(OUT / name, "w", **prof) as dst:
        dst.write(mosaic)
    return used


def main():
    OUT.mkdir(exist_ok=True)
    dem = fetch_dem_utm(utm_box(RENDER_LL, 8.0), 8.0, OUT / "mono_dem_8m.tif")
    hor = fetch_dem_utm(utm_box(HORIZON_LL, 60.0), 60.0, OUT / "horizon_dem_60m.tif")
    year, ids = fetch_naip(RENDER_LL, 5.0, "mono_naip_5m.tif")
    s2 = fetch_s2(HORIZON_LL, 60.0, "horizon_s2_60m.tif")
    (OUT / "sources.json").write_text(json.dumps({
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dem": {"service": IMAGESERVER, "note": "USGS 3DEP best-available (1 m lidar: CA_SierraNevada_B22, "
                "CA_YosemiteNP_2019)", "files": [dem.name, hor.name]},
        "naip": {"collection": "naip (Microsoft Planetary Computer)", "year": year, "items": ids},
        "sentinel2": {"collection": "sentinel-2-l2a (Microsoft Planetary Computer)", "items": s2},
        "crs": UTM, "render_extent_lonlat": RENDER_LL, "horizon_extent_lonlat": HORIZON_LL,
    }, indent=1))


if __name__ == "__main__":
    main()
