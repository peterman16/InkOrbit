# Architecture

The Architect agent keeps this file current. Update it in the same PR as any
structural change.

## The pipeline

```
Elements / arrays ──draw──▶ Scene ──layout──▶ Drawing ──renderer──▶ image
   data units              data units      page mm, plain ink      matplotlib (for now)
```

- **Scene** (`scene.py`): backend-neutral list of items (lines with a role, fills,
  arrowheads, markers, text) in data coordinates.
- **Layout** (`layout.py`): fits the scene on a `Page`, resolves every physical size, and
  turns text, arrows, markers and fill patterns into plain ink: strokes, solid polygons
  and dots. Label placement (`labels.py`) runs here, on the whole page.
- **Renderer** (`render/`): draws plain ink. Adding a backend means implementing that
  one small `Renderer` protocol.

Dependencies point one way: `draw` → `scene` → `layout` → `render`. `style` is read by
all of them and imports none of them. Nothing in the core imports an adapter.

## Module map and ownership

| Module | What lives there | Owner |
|---|---|---|
| `style.py` | Pens, line roles, lettering and pattern sizes | render-engineer |
| `lettering.py` | Hershey text to strokes | render-engineer |
| `patterns.py` | Stipple and hatch generation | render-engineer |
| `layout.py` | Page fitting, size resolution, ink conversion | render-engineer |
| `labels.py` | Automatic label placement (from step 2) | render-engineer |
| `render/` | Backends | render-engineer |
| `scene.py` | Scene items and the `Scene` builder | shared: Architect decides changes |
| `geometry.py` | Plane geometry: conics, arcs, rotations; ring arcs split near/far at the view plane; orthographic camera projection and the split of a polyline in front of and behind a globe | orbit-engineer |
| `elements.py` | `Elements` dataclass, frames (from step 2) | orbit-engineer |
| `draw.py` | Primitives: orbit, body, apsides, nodes, angles, vectors (from step 2) | orbit-engineer |
| `diagrams.py` / `plate.py` | Named diagrams built from `draw`; `Plate` | orbit-engineer |
| `surfaces.py` | Per-body surface data as lon/lat: gazetteer craters with seeded natural rims and shadow crescents, pre-generalised region rings (base and core levels) built offline by `tools/build_surfaces.py`, Earth coastlines, stylised gas-giant bands; Saturn's stylised ring bands (inner/outer radii in body radii, tone); data in `data/<body>_surface.npz` | orbit-engineer |
| `adapters.py` | poliastro/hapsira, Skyfield, astropy objects read by attribute; Skyfield imported lazily for ground tracks only | orbit-engineer |
| `printed.py` | Printed mode: rendered page to a 1-bit ink/paper bitmap (spread, grain, skew). Pure numpy. | render-engineer |

## Where new code goes

- A new *kind of thing on the page* (a new scene item) changes `scene` and `layout`
  together: Architect first.
- A new *picture of something orbital* is composed from existing `draw` primitives.
  If it needs a new primitive, add it to `draw.py`, not to the diagram.
- A new *look* (pen, pattern, size) is a `Style` field or a role, never a literal.
- Shared helpers go in the lowest module that both callers already import, usually
  `geometry` (data-space maths) or `patterns` (page-space maths).

## Decisions log

Add one line per structural decision: date, decision, reason.

- 2026-10: Backend-neutral scene with matplotlib as the first renderer, so a later
  SVG/Cairo backend only implements strokes, polygons and dots.
- 2026-10: Hershey vector fonts for lettering, so text is inked like every other line.
- 2026-10: No propagation in the core; adapters convert library objects to `Elements`
  or arrays.
- 2026-10-16: Printed mode is a post-render step: `RenderedFigure.to_array` → `printed.apply` →
  1-bit bitmap. Renderers only rasterise; `Plate` orchestrates. Strictly two colours, seeded
  (default 0); SVG/PDF embed the bitmap. numpy only, no direct Pillow import.
- 2026-10-17: The globe is oriented from the orbit's own frame (`geometry.orbit_view_rotation`),
  longitude 0 at the vernal equinox since there is no epoch. Bodies stay to scale; orbits within
  10% of the body radius drop their arrowheads instead.
- 2026-10-20: Surface features replace coastlines. `draw.body(body=, surface=)` never assumes Earth (KAN-18). One SURFACE role. Body-fixed longitude 0 sits at the globe's longitude 0; spin rotates. Gazetteer centre+diameter rings; gas-giant bands are constants; no new deps.
- 2026-10-21: Rocky globes get natural craters (seeded low-harmonic rims, smaller craters cut larger rims, west-lit shadow stipple; mons only on Mars and Venus) and graded-stipple regions. Regions are generalised offline (blur + contour; Moon from USGS mare outlines, Mercury/Venus from gazetteer plains; none on Mars yet, KAN-28) and stored as base/core rings; no runtime contourpy. Fills with holes are single even-odd rings bridged by `geometry.with_holes`; no Scene/Fill API change. Regions and shadows are limb-clipped on the sphere (`geometry.ring_polygons`). Stipple sizes are `Style` fields (custom Style not yet honoured, KAN-37); `surface=True` is the only switch.
- 2026-10-21: Saturn's rings are drawn by `draw.body` from `surfaces.rings` with `surface=True`, in the equatorial plane with the globe's rotation: the far half-bands before the globe, a paper knock-out of the disc, the globe, then the near half-bands over paper knock-outs. No new scene item. Ring shadows (on the globe and on the rings) are deferred to a v2.
- 2026-10-29: Oblique view (KAN-44): `view=(elevation, azimuth)` (radians, inertial/equatorial frame, azimuth from the vernal equinox) on `diagrams.orbit/orbits/hohmann` and the draw primitives, not on `Plate`. Projection is done in `draw` with `geometry` maths (camera = `view_rotation(elev, az)`, perifocal→inertial = `orbit_view_rotation(...).T`); the scene stays 2D, no new scene item. `geometry.split_behind` splits projected lines at the limb: the parts behind the disc are drawn in the HIDDEN role, the parts in front over the globe, with no knock-out. `view=None` keeps today's face-on path, so baselines are unchanged. Angle arcs, vectors and axes are left out in a view; reference plane, Ω/i arcs and ring occlusion of the orbit are deferred.
