# inkorbit

Orbital diagrams in the style of 1960s black-and-white technical print: technical-pen
line weights, stippled night sides, drafting arrowheads and single-stroke lettering.

![Proof of style](docs/images/proof_of_style.png)

> Early days. This is the foundation (step 1 of the plan): the drawing model, the
> look and the matplotlib renderer. Ready-made diagrams and adapters for poliastro,
> hapsira, Skyfield and astropy come next.

## Install (development)

```bash
uv sync
uv run python examples/proof_of_style.py
uv run pytest
```

## How it fits together

```
Scene  ──layout──▶  Drawing  ──renderer──▶  image
data coordinates    page millimetres       matplotlib (for now)
```

- **`Scene`**: what to draw, in your data units (km, Earth radii...). Lines carry a
  *role* (`orbit`, `transfer`, `apse`, `construction`, `grid`...) instead of a colour or width.
- **`Style`**: decides how each role looks: technical pen widths in mm, dash patterns,
  lettering, arrowheads, stipple and hatch density.
- **`layout`**: puts the scene on a page, resolves every physical size and turns text,
  arrows, markers and fill patterns into plain ink: strokes, solid polygons and dots.
- **Renderers** draw only those three things, so adding a backend besides matplotlib
  is small.

```python
import inkorbit as ink
from inkorbit import geometry as geo, style as st

scene = ink.Scene(caption="Figure 1  A transfer orbit")
scene.line(geo.circle((0, 0), 1.0), st.OUTLINE)
scene.line(geo.ellipse(2.0, 0.5), st.ORBIT, arrows=(0.3, 0.8))
scene.text("perigee", (1.0, 0), offset_mm=(1.5, -3))

fig = scene.render(page="figure")      # 160 x 120 mm
fig.savefig("transfer.png")            # 300 dpi
fig.data_ax.plot(...)                  # keep drawing with matplotlib, in data units
```

In Jupyter a `Scene` (or a rendered figure) displays itself as the last line of a cell.

## Credits

Lettering uses the Hershey vector fonts, created by Dr. A. V. Hershey at the U.S.
Naval Weapons Laboratory, via the MIT-licensed
[Hershey-Fonts](https://github.com/apshu/HersheyFonts) package.
