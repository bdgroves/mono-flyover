# Mono Basin flyover

A forge3d flight on a late-September morning: from Tioga Pass down Lee Vining Canyon, out over Mono Lake past Negit and Paoha islands, low over the South Tufa towers, over Panum Crater and south along the Mono Craters, settling on the craters with the Sierra Nevada's eastern escarpment behind them.

Work in progress.

## How it's made

- **Terrain:** USGS 3DEP 1 m lidar (CA_SierraNevada_B22 and CA_YosemiteNP_2019 cover the whole flight).
- **Colour:** USDA NAIP aerial photography, 0.6 m near the camera. The distant skyline uses Sentinel-2.
- **Nested layers:** forge3d's viewer draws at most about 2,048 vertices across a terrain, which over the full 36 km region is one point every ~18 m. So every frame is drawn in four layers, each under that cap, and composited finest-first:
  - a ~170 km horizon at ~85 m
  - the region at ~18 m
  - a mid window around what the camera sees within 4.5 km, at a few metres
  - a near window within 1.3 km, at 1–2 m, textured with 0.6 m photography
- **Windows:** they move with the flight. Each render chunk fetches its own windows for its stretch of the path.
- **Sun:** placed with forge3d's solar calculator for 26 September 2026, 8:45 am PDT.

The engine comes from [SOLSTICE](https://github.com/bdgroves/solstice), and the flight planner from the Yosemite Valley flyover.

## Run it

On GitHub Actions:

- **Prep render data:** fetches the base data once, onto the `prep-data` branch.
- **Render with forge3d:**
  - `mode=path` draws the flight map.
  - `mode=stills` renders a few frames.
  - `mode=flyover` renders the whole flight in parallel chunks and commits `renders/mono-flyover.mp4`.

Locally (pixi):

```powershell
pixi install
pixi run data      # base data into prep/
pixi run path      # flight map + timing -> renders/flight_map.png
pixi run stills    # a few frames along the flight
pixi run render    # every frame, chunk by chunk (fetches the 1 m windows as it goes)
pixi run encode
```

Edit `KEYS` in `tools/render.py` to change the flight. Each keyframe has:
- an eye point and an aim point (latitude, longitude, altitude in metres)
- the speed through it
- the lens

The timing follows from the speeds. The path is lifted wherever it would pass within 90 m of the ground.
