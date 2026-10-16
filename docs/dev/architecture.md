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
| `geometry.py` | Plane geometry: conics, arcs, rotations | orbit-engineer |
| `elements.py` | `Elements` dataclass, frames (from step 2) | orbit-engineer |
| `draw.py` | Primitives: orbit, body, apsides, nodes, angles, vectors (from step 2) | orbit-engineer |
| diagrams / `Plate` | Named diagrams built from `draw` (planned, step 3) | orbit-engineer |
| adapters | poliastro/hapsira, Skyfield, astropy as optional extras (planned, step 3) | orbit-engineer |
| printed mode | Post-processing of the rendered image (planned, step 4) | render-engineer |

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
