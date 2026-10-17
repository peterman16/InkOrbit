"""Printed mode: make a clean raster look like offset print scanned from paper.

This runs on the finished raster, after a renderer has drawn it, and returns a
strictly two-colour bitmap. Every effect (ink spread, rough edges) happens on
a coverage field *before* the final threshold, so there is never a grey pixel.
Drawing config (pens, lettering) stays in ``Style``; only the amount of print
damage lives in :class:`Printed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Printed:
    """How worn the print looks. Sizes are millimetres on the page, so the result
    does not depend on the dpi it is rasterised at."""

    ink_spread_mm: float = 0.035  # blur of the ink into the paper (Gaussian sigma)
    edge_noise: float = 0.12  # amplitude of the rough-edge noise, as a fraction of full ink
    edge_grain_mm: float = 0.06  # correlation length of that noise (size of the bumps)
    # Real ink spreads outwards: lowering the threshold from half-tone grows every mark by a fraction
    # of the spread, so small dots and hairlines keep at least their drawn size instead of eroding.
    ink_gain: float = 0.1
    # Scan geometry: the sheet lies slightly crooked and the plate is slightly out of true.
    skew_deg: float = 0.12  # rotation of the whole page about its centre
    warp_mm: float = 0.25  # amplitude of the smooth displacement
    warp_wavelength_mm: float = 90.0
    # Grain. Densities are expected counts per mm² of page, sizes are the side of the square blob.
    dropout_density: float = 0.15  # pinholes where ink failed to take
    dropout_mm: float = 0.07
    speck_density: float = 0.006  # stray ink on bare paper
    speck_mm: float = 0.09


DEFAULT_PRINTED = Printed()


def apply(rgba: np.ndarray, dpi: float, params: Printed = DEFAULT_PRINTED, seed: int = 0) -> np.ndarray:
    """Return a bool array, True where there is ink, from an ``(h, w, 4)`` uint8 render."""
    rng = np.random.default_rng(seed)
    px_per_mm = dpi / MM_PER_INCH
    # Darkness of the render: 1 where fully inked, 0 on bare paper (alpha is not used).
    coverage = 1.0 - rgba[..., :3].mean(axis=2, dtype=np.float32) / 255.0
    coverage = _scan(coverage, px_per_mm, params, rng)
    coverage = _smooth(coverage, _gaussian(params.ink_spread_mm * px_per_mm))
    # Roughen only near ink: on bare paper the lowered threshold would turn noise peaks into stray specks.
    near_ink = coverage > 0.01
    coverage += params.edge_noise * _noise(coverage.shape, params.edge_grain_mm * px_per_mm, rng) * near_ink
    coverage += _blobs(coverage.shape, params.speck_density, params.speck_mm, px_per_mm, rng)
    coverage -= _blobs(coverage.shape, params.dropout_density, params.dropout_mm, px_per_mm, rng)
    return coverage > 0.5 - params.ink_gain


def _gaussian(sigma_px: float) -> np.ndarray:
    """Normalised Gaussian kernel; a single tap (no smoothing) below a tenth of a pixel."""
    if sigma_px < 0.1:
        return np.ones(1, dtype=np.float32)
    x = np.arange(-int(np.ceil(3 * sigma_px)), int(np.ceil(3 * sigma_px)) + 1)
    k = np.exp(-0.5 * (x / sigma_px) ** 2).astype(np.float32)
    return k / k.sum()


def _smooth(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable convolution with edge padding, as a sum of shifted slices (fast for small kernels)."""
    r = len(kernel) // 2
    for axis in (0, 1):
        padded = np.pad(image, [(r, r) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        n = image.shape[axis]
        image = sum(w * np.take(padded, np.arange(i, i + n), axis=axis) for i, w in enumerate(kernel))
    return image


def _noise(shape: tuple[int, ...], corr_px: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance noise whose bumps are about ``corr_px`` wide."""
    kernel = _gaussian(corr_px)
    noise = _smooth(rng.standard_normal(shape, dtype=np.float32), kernel)
    # Smoothing shrinks the variance by sum(k^2) per axis; undo it so edge_noise means what it says.
    return noise / np.sum(kernel**2)


def _blobs(shape: tuple[int, ...], density_per_mm2: float, size_mm: float, px_per_mm: float, rng) -> np.ndarray:
    """A 0/1 field with square blobs of ``size_mm`` scattered at ``density_per_mm2``."""
    seeds = rng.random(shape, dtype=np.float32) < density_per_mm2 / px_per_mm**2
    side = max(1, round(size_mm * px_per_mm))
    # Spread each seed over a side x side square (the box is centred, so shifted by half a pixel at most).
    return (_smooth(seeds.astype(np.float32), np.ones(side, np.float32)) > 0).astype(np.float32)


def _scan(coverage: np.ndarray, px_per_mm: float, params: Printed, rng: np.random.Generator) -> np.ndarray:
    """Rotate the page about its centre and bend it with a gentle sinusoidal warp (bilinear resampling)."""
    h, w = coverage.shape
    phases = rng.uniform(0, 2 * np.pi, size=2)  # always drawn, so a seed means the same warp whatever the amounts
    theta = np.radians(params.skew_deg)
    if theta == 0 and params.warp_mm == 0:
        return coverage
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    # Where, in the flat page, does each pixel of the scan come from?
    sx = cx + np.cos(theta) * (x - cx) - np.sin(theta) * (y - cy)
    sy = cy + np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy)
    k = 2 * np.pi / (params.warp_wavelength_mm * px_per_mm)
    sx += params.warp_mm * px_per_mm * np.sin(k * y + phases[0])
    sy += params.warp_mm * px_per_mm * np.sin(k * x + phases[1])
    # Bilinear lookup in a copy padded with paper, so what rotates in from outside is blank.
    padded = np.pad(coverage, 1)
    sx = np.clip(sx + 1, 0, w + 0.999)
    sy = np.clip(sy + 1, 0, h + 0.999)
    x0, y0 = sx.astype(int), sy.astype(int)
    fx, fy = sx - x0, sy - y0
    top = padded[y0, x0] * (1 - fx) + padded[y0, x0 + 1] * fx
    bottom = padded[y0 + 1, x0] * (1 - fx) + padded[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy
