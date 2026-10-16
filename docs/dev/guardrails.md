# Guardrails

Hard rules. If a task seems to need breaking one, stop and ask Peter through the Lead.

## The look is the product

- **Black ink on white paper only.** No colour, no grey, no alpha blending. Tone comes
  from hatching and stipple, never shading.
- **Line roles, not line styles.** Code says what a line *is* (`style.ORBIT`,
  `style.HIDDEN`...). Widths, dashes and sizes come from `Style`. Never hard-code a
  colour, linewidth or dash pattern outside `style.py`.
- **Physical units.** Anything with a size on the page is in millimetres, resolved by
  `layout`. Geometry is in data units. Don't mix them.
- **Lettering is Hershey strokes.** No system fonts and no matplotlib text, so every
  backend inks text the same way.
- **Renderers stay dumb.** A renderer draws strokes, solid polygons and dots in page
  millimetres, nothing else. New visual features are built in `scene`/`layout`/`draw`,
  not in a renderer.

## Image baselines

- `tests/baseline/*.png` are the reference look. A failing image test means the look
  changed: find out why before touching anything.
- Regenerate a baseline only when the change is intended, and say so in the PR with a
  before/after image. Never regenerate to make CI pass.

## Scope and dependencies

- No orbit propagation in the core. inkorbit draws; poliastro/hapsira, Skyfield and
  astropy compute. Their adapters are optional extras and the core never imports them.
- New runtime dependencies need Peter's approval. Core dependencies today: numpy,
  matplotlib, Hershey-Fonts.
- Public API changes (new or renamed names in `inkorbit.__all__`, changed signatures)
  go past the Architect and are called out in the PR.
- Python 3.10+. CI runs 3.10 to 3.13.

## Process

- Never push to `main`. Every change goes through a PR with green CI.
- Never skip, delete or loosen a test (including image tolerances) to get green.
- Don't edit files outside the task's area without saying why in the PR.
- If unsure about orbital mechanics, a reference figure, or what Peter wants: say so
  and ask. Don't invent sources.
