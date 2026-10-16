"""Build inkorbit's bundled map data from Natural Earth 1:110m GeoJSON (public domain).

Usage:
    curl -O https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson
    curl -O https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson
    python tools/build_coastlines.py ne_110m_coastline.geojson ne_110m_land.geojson src/inkorbit/data/earth_110m.npz
"""

import json
import sys

import numpy as np


def _pack(rings):
    points = np.concatenate([np.asarray(r, dtype=np.float32) for r in rings])
    offsets = np.cumsum([0] + [len(r) for r in rings]).astype(np.int32)
    return points, offsets


def main(coastline_path, land_path, out_path):
    with open(coastline_path) as f:
        coast = [feat["geometry"]["coordinates"] for feat in json.load(f)["features"]]
    with open(land_path) as f:
        land = []
        for feat in json.load(f)["features"]:
            geom = feat["geometry"]
            polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            land += [polygon[0] for polygon in polygons]  # outer rings only
    coast_pts, coast_off = _pack(coast)
    land_pts, land_off = _pack(land)
    np.savez_compressed(
        out_path,
        coastline_points=coast_pts,
        coastline_offsets=coast_off,
        land_points=land_pts,
        land_offsets=land_off,
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])
