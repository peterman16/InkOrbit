"""Build inkorbit's bundled surface feature data from the USGS Gazetteer of Planetary Nomenclature.

Source: IAU WGPSN / USGS Astrogeology Science Center, Gazetteer of Planetary Nomenclature,
https://planetarynames.wr.usgs.gov/GIS_Downloads (feature centre points, "nomenclature_center_pts").
The metadata shipped in each zip states: "<useconst>Public domain.</useconst>".

Usage (one body at a time; BODY is MOON, MERCURY, VENUS or MARS):
    curl -O https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/BODY_nomenclature_center_pts.zip
    python tools/build_surfaces.py BODY_nomenclature_center_pts.zip src/inkorbit/data/body_surface.npz

Filter: IAU-approved features whose type is a crater (on the Moon also lettered satellite craters), mare or
oceanus, mons or planitia (on Venus also terra and regio), with diameter at or above the body's
MIN_DIAMETER_KM. The minimums keep each file under ~50 KB. Output arrays, sorted largest first: lat and lon
in degrees (planetocentric, east longitude in [-180, 180)), diameter_km, and kind, an index into the kinds
array.

Regions step (run after the step above, which rewrites the file): adds dark-region rings, pre-generalised so
the library only fills them. Needs contourpy (and pyshp for the Moon), used only here:
    uv run --with contourpy --with pyshp python tools/build_surfaces.py regions moon_surface.npz GeoUnits.shp
    uv run --with contourpy python tools/build_surfaces.py regions mercury_surface.npz
Moon: the Em, Im1 and Im2 (mare) units of the Unified Geologic Map of the Moon, 1:5M, 2020 (Fortezzo, Spudis and
Harrel), https://asc-astropedia.s3.us-west-2.amazonaws.com/Moon/Geology/Unified_Geologic_Map_of_the_Moon_GIS_v2.zip,
Unified_Geologic_Map_of_the_Moon_GIS/Lunar_GIS/Shapefiles/GeoUnits.shp (unzip the .shp, .shx, .dbf next to each
other). Astropedia lists its use constraints as "CC0, but please cite authors". The polygons are rasterised
even-odd, so islands inside a mare punch holes. Mercury and Venus have no such map: their mare and planitia
gazetteer features become discs with seeded wobbly edges. Mars gets no regions (KAN-28).
The raster (0.5 deg) is blurred, wrapping in longitude; contour level 0.5 of a 1.5 deg blur is the base ring and
level 0.8 of a 3 deg blur the core ring, so the core is the heart of each region. Rings under 12 deg^2 are
dropped and the rest smoothed to look hand-traced. Output: region_points (lon, lat), region_offsets, and
region_level (0 base, 1 core) per ring, packed like earth_110m.npz. Rings are closed; those meeting the map edge
(lon +-180, lat +-90) run along it, with points every degree so no segment jumps across the map.
"""

import itertools
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np

MIN_DIAMETER_KM = {"MOON": 20.0, "MERCURY": 1.0, "VENUS": 1.0, "MARS": 1.0}

KINDS = ("crater", "mare", "mons", "planitia", "terra", "regio")
_KIND_OF_TYPE = {
    "Crater, craters": "crater",
    "Satellite Feature": "crater",  # lettered craters next to a named one, e.g. Tycho A
    "Mare, maria": "mare",
    "Oceanus, oceani": "mare",
    "Mons, montes": "mons",
    "Planitia, planitiae": "planitia",
}
# Venus's best-known features (Ishtar, Aphrodite) are terrae and regiones; elsewhere they would only add clutter.
_EXTRA_KIND_OF_TYPE = {"VENUS": {"Terra, terrae": "terra", "Regio, regiones": "regio"}}


def _dbf_records(data):
    """Yield each live record of a dBASE III table as a dict of stripped strings."""
    count, header_len, record_len = struct.unpack("<4xIHH", data[:12])
    fields, pos = [], 32
    while data[pos] != 0x0D:
        fields.append((data[pos : pos + 11].rstrip(b"\0").decode(), data[pos + 16]))
        pos += 32
    for i in range(count):
        record = data[header_len + i * record_len : header_len + (i + 1) * record_len]
        if record[:1] == b"*":  # deleted
            continue
        values, at = {}, 1
        for name, width in fields:
            values[name] = record[at : at + width].decode("utf-8", "replace").strip()
            at += width
        yield values


def main(zip_path, out_path):
    with zipfile.ZipFile(zip_path) as z:
        (dbf_name,) = [n for n in z.namelist() if n.endswith(".dbf")]
        body = dbf_name.split("_")[0].upper()
        records = list(_dbf_records(z.read(dbf_name)))
    kind_of_type = _KIND_OF_TYPE | _EXTRA_KIND_OF_TYPE.get(body, {})
    rows = [
        (float(r["center_lat"]), float(r["center_lon"]), float(r["diameter"]), KINDS.index(kind_of_type[r["type"]]))
        for r in records
        if r["approval"] == "Adopted by IAU"
        and r["type"] in kind_of_type
        and float(r["diameter"]) >= MIN_DIAMETER_KM[body]
    ]
    rows.sort(key=lambda row: -row[2])
    lat, lon, diameter, kind = (np.array(col) for col in zip(*rows, strict=True))
    np.savez_compressed(
        out_path,
        lat=lat.astype(np.float32),
        lon=((lon + 180.0) % 360.0 - 180.0).astype(np.float32),
        diameter_km=diameter.astype(np.float32),
        kind=kind.astype(np.uint8),
        kinds=np.array(KINDS),
    )
    print(f"{body}: {len(rows)} features -> {out_path}")


STEP = 0.5  # raster step, degrees
MOON_DEG_PER_M = 180.0 / (np.pi * 1737400.0)  # GeoUnits.shp is equidistant cylindrical, in metres
LEVELS = ((0.5, 1.5), (0.8, 3.0))  # (contour level, blur sigma in degrees): base, core
MIN_AREA = 12.0  # deg^2, longitude scaled by cos(lat)


def _blur(field, sigma):
    """Gaussian blur of a (lat, lon) raster, wrapping in longitude and repeating the edge rows at the poles."""
    k = np.arange(-3 * sigma, 3 * sigma + STEP, STEP)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    n = len(w) // 2
    field = np.pad(np.pad(field, ((0, 0), (n, n)), mode="wrap"), ((n, n), (0, 0)), mode="edge")
    for axis in (0, 1):
        field = np.apply_along_axis(np.convolve, axis, field, w, "same")
    return field[n:-n, n:-n]


def _unit_vectors(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def _mask(body, shp, grid):
    """Boolean (lat, lon) raster of the dark regions: mare polygons (Moon) or wobbly discs (the others)."""
    from matplotlib.path import Path as Polygon

    from inkorbit import surfaces
    from inkorbit.bodies import get_body

    lon, lat = np.meshgrid(*grid)
    points = np.column_stack([lon.ravel(), lat.ravel()])
    mask = np.zeros(len(points), bool)
    if body == "moon":
        import shapefile

        for item in shapefile.Reader(shp).iterShapeRecords():
            if item.record["FIRST_Unit"] not in ("Em", "Im1", "Im2"):
                continue
            shape = np.array(item.shape.points) * MOON_DEG_PER_M
            for a, b in itertools.pairwise([*item.shape.parts, len(shape)]):
                ring = shape[a:b]
                lo, hi = ring.min(0), ring.max(0)
                near = np.all((points >= lo) & (points <= hi), axis=1)
                mask[near] ^= Polygon(ring).contains_points(points[near])  # even-odd: holes and islands
        return mask.reshape(lon.shape)
    # Mare and planitia caps (centre and radius), each edge wobbled by a few seeded harmonics
    unit = _unit_vectors(points[:, 1], points[:, 0])
    for i, (clat, clon, radius) in enumerate(surfaces._caps(get_body(body), (1, 3), (0.03, 0.6))):
        centre = _unit_vectors(clat, clon)
        near = np.degrees(np.arccos(np.clip(unit @ centre, -1, 1)))
        sel = near < 1.4 * radius
        east = np.cross([0, 0, 1], centre)
        east /= np.linalg.norm(east) + 1e-12
        bearing = np.arctan2(unit[sel] @ east, unit[sel] @ np.cross(centre, east))
        rng = np.random.default_rng(zlib.crc32(f"{body}{i}".encode()))
        wobble = sum(rng.uniform(0.04, 0.1) * np.cos(k * bearing + rng.uniform(0, 2 * np.pi)) for k in (2, 3, 4, 6, 9))
        mask[np.flatnonzero(sel)[near[sel] < radius * (1 + wobble)]] = True
    return mask.reshape(lon.shape)


def _smooth(ring, k=15):
    """Circular moving average of an open ring (first point not repeated), then every third point (the average
    has no detail left). Points on the map edge stay on it, and the ends of edge runs are kept."""
    pad = np.concatenate([ring[-k:], ring, ring[:k]])
    out = np.column_stack([np.convolve(pad[:, i], np.ones(k) / k, "same")[k:-k] for i in (0, 1)])
    edge = np.abs(ring) >= [180, 90]
    out[edge] = ring[edge]
    on = edge.any(1)
    return out[(np.arange(len(ring)) % 3 == 0) | (on & ~(np.roll(on, 1) & np.roll(on, -1)))]


def _rings(lon, lat, field, level):
    """Outer rings of the region where ``field`` exceeds ``level``: closed, smoothed, edge runs densified."""
    import contourpy

    generator = contourpy.contour_generator(lon, lat, field, fill_type=contourpy.FillType.OuterOffset)
    out = []
    for polygon, offsets in zip(*generator.filled(level, 2.0), strict=True):
        ring = polygon[offsets[0] : offsets[1] - 1]  # outer ring only, without the repeated first point
        x, y = ring[:, 0] * np.cos(np.radians(ring[:, 1])), ring[:, 1]
        if 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < MIN_AREA:
            continue
        ahead = np.roll(ring, -1, 0)
        steps = np.maximum(1, np.ceil(np.abs(ahead - ring).max(1))).astype(int)  # at most 1 degree apart
        ring = np.concatenate(
            [a + (b - a) * np.arange(n)[:, None] / n for a, b, n in zip(ring, ahead, steps, strict=True)]
        )
        ring = _smooth(ring)
        out.append(np.vstack([ring, ring[:1]]))
    return out


def regions(npz_path, shp=None):
    """Add region_points, region_offsets and region_level to a body's surface file (see the module docstring)."""
    body = Path(npz_path).name.split("_")[0]
    with np.load(npz_path) as data:
        arrays = {k: data[k] for k in data.files if not k.startswith("region_")}
    lon, lat = np.arange(-180, 180, STEP), np.arange(-90, 90 + STEP, STEP)
    mask = _mask(body, shp, (lon, lat)).astype(float)
    lon = np.append(lon, 180.0)  # the seam column repeats -180, so rings reach the edge and close along it
    rings, level = [], []
    for i, (threshold, sigma) in enumerate(LEVELS):
        field = _blur(mask, sigma)
        found = _rings(lon, lat, np.hstack([field, field[:, :1]]), threshold)
        rings += found
        level += [i] * len(found)
    arrays["region_points"] = np.concatenate(rings).astype(np.float32)
    arrays["region_offsets"] = np.cumsum([0] + [len(r) for r in rings]).astype(np.int32)
    arrays["region_level"] = np.array(level, np.int8)
    np.savez_compressed(npz_path, **arrays)
    print(f"{body}: {level.count(0)} base + {level.count(1)} core rings -> {npz_path}")


if __name__ == "__main__":
    if sys.argv[1] == "regions":
        regions(*sys.argv[2:4])
    else:
        main(*sys.argv[1:3])
