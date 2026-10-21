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
"""

import struct
import sys
import zipfile

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


if __name__ == "__main__":
    main(*sys.argv[1:3])
