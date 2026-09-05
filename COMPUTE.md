# Compute Costs

All figures measured on a single A100 (Colab High-RAM). These explain why the benchmark is 40 problems and why distillation stopped at 50.

## Inference: the compression-floor grid

Full sweep, 2 block sizes × 5 schedules × 40 problems, greedy, budget 1536: **52 minutes**.

Per-cell wall-clock (40 problems each) shows the cost structure clearly:

| tokens/step | block 16 | block 32 |
|---|---|---|
| 1 | 13.2 min | 13.6 min |
| 2 | 6.1 min | 6.6 min |
| 4 | 3.6 min | 3.5 min |
| 8 | 1.7 min | 2.0 min |
| 16 | 0.7 min | 0.7 min |

Cost halves as tokens/step doubles, and is nearly identical across block sizes at matched tokens/step. This is what makes the block-16 vs block-32 comparison in the paper compute-matched: a generation of length L costs L/t forward passes regardless of block partitioning.

## Training: SFT curricula

Forward-pass count per stage is `n_problems × epochs × (max_target_tokens / tokens_per_step)`:

| Stage | tokens/step | forwards | ~hours |
|---|---|---|---|
| S1 | 4 | 96,000 | 1.5 |
| S2 | 2 | 192,000 | 2.9 |
| S3 | 1 | 384,000 | 5.9 |
| **Total** | | **672,000** | **~10.3** |

Plus ~2.5 h benchmarking (3 checkpoints × 2 schedules). Both arms cost the same: the masked-only arm trains on 3× the problems but 1/3 as many per stage.

`max_target_tokens` is the largest lever. Halving it to 192 roughly halves every stage.

## Training: OPSD

The cost that surprised us, stated as arithmetic so nobody repeats the mistake. At `gen_budget=1536` (48 blocks per problem), with a 2 tok/step student and 1 tok/step teacher:

| Segment | Forwards per problem | Gradient? |
|---|---|---|
| Student rollout | 16 × 48 = 768 | yes (+48 backwards) |
| Teacher scoring | 32 × 48 = 1,536 | no |
| **Total** | **~2,350 forward-equivalents** | |

An 8 tok/step inference pass over the same problem is 192 forwards. **A training pass therefore costs roughly 12× an inference pass**, before any pass multiplier. "Training is just inference twice" undercounts by an order of magnitude.

Mitigations implemented in notebook 04:
- Teacher sub-step states are **batched** (`teacher_batch`), since they are all known once the student commits a block. Raising this toward 32 runs a whole block's teacher pass in one forward.
- Per-problem `student / teacher / backward` timers print so the dominant segment is visible rather than guessed.

There is no KV-cache. Every sub-step reprocesses the full growing context, which is why cost grows superlinearly with budget. Adding a prefix KV-cache is the single highest-value optimization left undone here.

## Why n = 40

A single schedule sweep is 6 decoding configurations × 40 problems. One cell at 1 token/step exceeds six GPU-hours at the SFT benchmark budget. We ran that sweep across the base model at two block sizes and multiple fine-tuned checkpoints, plus screening at several sampling settings. At n=200 the same experimental design would not have fit in the available compute.

The honest cost: per-cell accuracy carries binomial error near ±15 points. The paper treats individual cells as large-effect observations rather than precise estimates, and every claim rests on a pattern across many cells rather than one number.
