# Reproduction Guide

Run the notebooks in numerical order. Each stage consumes the previous stage's outputs, so out-of-order execution will fail with a missing-file assertion rather than silently producing wrong numbers.

## Prerequisites

- **Notebooks 01–04:** Google Colab with an A100 (High-RAM). Roughly 40 GB VRAM is needed for 4B + LoRA at `gen_budget=1536`.
- **Notebook 05:** a local machine with a CUDA GPU, ~40 GB VRAM, and Jupyter. No Colab or Drive dependency.
- A Hugging Face account is not required; all checkpoints used are public.

### The transformers version constraint

`modeling_sdar.py` (shipped with the SDAR checkpoints via `trust_remote_code=True`) requires a transformers 4.5x release. Each notebook's setup cell removes `torchao` **before** any transformers import and pins the version. This ordering matters: importing transformers first leaves a stale module in memory and the pin silently fails.

**If the setup cell changes the installed version, restart the runtime and re-run from the top.** The notebooks print an explicit instruction when this happens.

## Stage 1 — Compression floor grid (`01_compression_floor_grid.ipynb`)

Sweeps the base model across block sizes {16, 32} × tokens-per-step {1, 2, 4, 8, 16} on the fixed 40 problems, greedy decoding, `gen_budget=1536`.

- **Runtime:** ~52 minutes on an A100 for the full grid.
- **Outputs:** `grid_results_full.json`, `grid_aggregate_full.csv`, `grid_per_problem_full.csv`, and the problem set itself.
- **First run creates `problem_set.json`** by sampling DeepMath-103K stratified 20 easy / 10 medium / 10 hard at a fixed seed. **Copy this file to `data/benchmark_problems.json` and reuse it for every later stage** — this is what makes the numbers comparable across notebooks. A copy of ours is already in `data/`; use it to reproduce our exact numbers rather than resampling.

Note: block 4 (`JetLM/SDAR-4B-Chat`) is attempted by the notebook and **fails to load** — that checkpoint does not ship `fused_linear_diffusion_cross_entropy.py`. The notebook records a skip and continues. This is expected, not a bug in this code.

## Stage 2 — Dense SFT curriculum (`02_sft_dense_curriculum.ipynb`)

Three-stage curriculum (4 → 2 → 1 tokens/step), `supervise_all=True`, 500 problems reused across all three stages, 2 epochs per stage.

- **Runtime:** ~10 hours training + ~2.5 hours benchmarking on an A100.
- **Requires:** `data_partitions.json` in the Drive data root (the notebook's own data cell builds it on first run).
- **Outputs:** LoRA adapters per stage (`ckpt/s1_t4`, `s2_t2`, `s3_t1`) and `results/curriculum_benchmark.json`.
- Resumable at sample granularity; re-running after a disconnect continues mid-stage.

## Stage 3 — Masked-only SFT curriculum (`03_sft_masked_only_curriculum.ipynb`)

The controlled counterpart. **The only deliberate objective change is `supervise_all=False`.** All hyperparameters are identical to Stage 2 (see [HYPERPARAMETERS.md](HYPERPARAMETERS.md)).

Two disclosed differences beyond the objective, both stated in the paper's limitations:
- **Data:** three disjoint 500-problem pools (1500 total) rather than one 500-problem pool reused three times. `data_partitions.json` only holds ~500 problems, so the notebook sources the shortfall directly from DeepMath-103K, stratified to match the existing tier mix.
- **Targets:** `<think>...</think>` reasoning traces are stripped from solutions before length-ranking, and the shortest surviving solution containing `\boxed{}` is used.

Set `cfg.bench_include_raw = True` if you want the base-model row re-measured rather than reusing ours.

- **Runtime:** same as Stage 2 (identical forward-pass count).
- **Outputs:** `results/normal_sft_benchmark.json`.

## Stage 4 — OPSD-1, privileged teacher (`04_opsd1_privileged_teacher.ipynb`)

On-policy self-distillation from **raw pre-SFT weights**, with the teacher conditioned on a DeepMath reference solution and decoding at 1 token/step against the student's 2.

Two-phase, chunked: phase 1 screens the student's own rollouts and keeps those reaching a `\boxed{}` answer; phase 2 trains on the accepted trajectories.

- **Critical setting:** `screen_temp=0.3`, `gen_budget=1536`. At temperature 0.9 with budget 768, yield is **zero** and no training occurs. This is the paper's precondition-1 result; it is not a bug, but it will waste your GPU hours if you change these values without reading Section 4.3.
- **Run §11 (the throughput probe) before committing to a full run.** It prints a per-problem `student / teacher / backward` breakdown and projects total hours.
- **Expected outcome:** the distillation loss collapses toward zero. This is the paper's precondition-2 result and is reproducible, not a failure of the implementation.
- **Runtime:** highly variable; the probe will tell you. Budget several hours.

## Stage 5 — OPSD-2, self-future teacher (`05_opsd2_dopsd_lookahead.ipynb`)

Local-Jupyter variant. Teacher and student share the same prompt and the same schedule; the teacher's only advantage is `future_reveal_k` positions revealed from later in the student's own reveal order.

**Run the null control first.** Set `cfg.future_reveal_k = 0` and run §11: the loss must collapse to ~0, because the teacher is then bit-identical to the student. If it does not, the KL positions are misaligned and the run is worthless. This check takes two minutes and is the highest-value thing to do before spending GPU time.

This notebook has not been run to completion at full scale. It is the construction the paper's evidence points toward, and it is provided for others to take further.

## Verifying you reproduced our numbers

```bash
python scripts/verify_results.py --grid /path/to/your/grid_aggregate_full.csv
```

Reports per-cell differences against `results/floor_grid_aggregate.csv`. Given n=40 and a single seed, differences within roughly ±0.10 on accuracy metrics are consistent with sampling noise; larger gaps indicate a genuine configuration difference worth tracking down.
