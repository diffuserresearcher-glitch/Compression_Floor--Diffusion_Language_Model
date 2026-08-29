# Two Evaluators, and Why Their Numbers Differ

This project used two evaluation harnesses. Both score the same 40 problems with the same prompts, but they use different stopping rules and different match criteria. **Numbers from the two harnesses are not comparable and are never mixed within one table or figure in the paper.**

## Harness A — compression-floor grid (notebook 01)

- Used for: base-model sweep, SFT-only schedule grid.
- Files: `results/floor_grid_*.csv`, `results/sft_dense_schedule_grid.csv`.
- Stopping: repetition-collapse early stop, plus EOS and budget.
- Budget: 1536.
- Reports: `boxed` (an answer was produced), `correct` (strict match), `repeat4`, plus structure metrics (`progress`, `redundancy`, `coverage`) and stop-reason fractions.
- Base model at block 32, t=1: **correct 0.375**.

## Harness B — SFT curriculum notebooks (notebooks 02, 03)

- Used for: the supervision-density comparison.
- Files: `results/sft_curriculum_benchmark.json`.
- Stopping: EOS and budget, no collapse detector.
- Budget: 1024.
- Reports: `strict` (normalized exact match) and `overall` (lenient: case-insensitive plus numeric equivalence via fraction evaluation), plus per-tier breakdown and `repeat4`.
- Base model at block 32, t=1: **overall 0.48**, strict 0.48.

## Why the base model scores 0.375 in one and 0.48 in the other

Three compounding differences, none of them a bug:

1. **Budget.** 1536 vs 1024 tokens. Different truncation behaviour on long solutions.
2. **Collapse detector.** Harness A halts a generation that starts looping, scoring it as a failure. Harness B lets it run to budget, where it may still recover and emit an answer.
3. **Match criterion.** Harness B's `overall` accepts numerically equivalent answers (e.g. `1/2` vs `0.5`); Harness A's `correct` is stricter.

Both numbers are correct for their own harness. The paper labels every table with which evaluator produced it, and the base-model row is reported separately under each.

## Rule for anyone extending this work

If you add a result, state which harness produced it, and do not place it in a table beside numbers from the other one. If you need a cross-harness comparison, re-run both arms under a single harness rather than adjusting numbers.
