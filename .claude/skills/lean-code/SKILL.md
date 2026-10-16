---
name: lean-code
description: Write the least code that meets the goal and reuse or extract shared code instead of duplicating it. Use for every coding task in inkorbit, and when reviewing a change for size or duplication.
---

# Lean code

The goal is the smallest diff that fully does the task, with no copy-paste.

## Before writing anything

1. **Restate the goal** in one sentence and the acceptance criteria. Anything not in
   them is out of scope.
2. **Search for what exists.** Grep for the concept (e.g. `arc`, `rotate`, `point_along`,
   `points_in_polygon`, a `Style` field) before writing a helper. The modules are small;
   skim the ones in your area.
3. **Prefer, in order:** using an existing function as is → passing it a new argument →
   generalising it slightly → writing something new.

## While writing

- No speculative features: no options, hooks, classes or config nobody asked for.
- No new abstraction for a single use. A plain function beats a class; a parameter
  beats a subclass.
- Use numpy vectorised operations instead of Python loops where it is shorter and clearer.
- Don't wrap a call just to rename it.
- Keep error handling to the public boundary; don't re-validate internally.
- Delete code your change makes dead, in the same PR.

## Shared code

- If you are about to write logic that already exists elsewhere, or that a second caller
  will obviously need, move it to one place and call it from both.
- Put it in the lowest module both callers already import (see "Where new code goes" in
  `docs/dev/architecture.md`). Don't create a `utils.py` grab-bag.
- Extract only real duplication (same logic, same reason to change), not code that just
  looks alike.
- If extracting would touch another engineer's area, report it to the Lead instead of
  doing it.

## Before handing back

Re-read your diff and ask:

- Could any added line be removed without failing a test or criterion?
- Did I duplicate anything that exists? (Grep for the core expression.)
- Is every new public name necessary?

Report the line count of the diff (`git diff --stat`) and anything you reused.
