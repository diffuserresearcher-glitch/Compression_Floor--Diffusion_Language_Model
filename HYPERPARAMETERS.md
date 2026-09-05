# Hyperparameters

All values below are read directly from the notebook config cells, not from memory. Where the two SFT arms differ, that is called out explicitly.

## Shared base configuration

| Setting | Value |
|---|---|
| Base model | `JetLM/SDAR-4B-Chat-b32` |
| Block size | 32 |
| Mask token id | 151669 |
| Dataset | `zwhe99/DeepMath-103K` |
| Benchmark | 40 problems, stratified 20 easy / 10 medium / 10 hard |
| Difficulty tiers | easy ≤ 4.0, medium ≤ 6.0, else hard (DeepMath `difficulty` field) |
| Precision | bfloat16 with autocast |
| Attention | `sdpa`, with a manual block-causal mask (see below) |

### The block-causal mask

`modeling_sdar.py` ships with `_update_causal_mask` commented out, so the stock HuggingFace forward applies **no** causal structure. Every notebook installs its own additive mask: bidirectional within each 32-token grid cell, causal across cells, where the grid is `absolute_position // block_size`.

Each notebook includes a sanity probe that asserts (a) perturbing a token in a later cell does not change logits in an earlier cell, and (b) it *does* change earlier positions within the same cell. If the probe fails, switch `attn_impl` to `"eager"` and re-run. Do not proceed past a failing probe: without the mask the model is not doing block diffusion at all.

## SFT curricula (notebooks 02 and 03)

Both arms use **identical** values for everything below. This is deliberate: the objective is the experiment.

| Hyperparameter | Value |
|---|---|
| LoRA rank (`r`) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | AdamW (`torch.optim.AdamW`) |
| Learning rate | 1e-5 |
| LR schedule | cosine with warmup (`get_cosine_schedule_with_warmup`) |
| Warmup steps | 10 |
| Batch size | 1 sequence per step |
| Gradient accumulation | 4 (effective batch size 4) |
| Gradient clipping | 1.0 |
| Epochs per stage | 2 |
| Stages | 3, at 4 → 2 → 1 tokens/step |
| Max target tokens | 384 |
| GPU count | 1 (single-device; no DDP or model sharding) |
| Benchmark budget | 1024 |
| Benchmark schedules | tokens/step ∈ {2, 1} |

### The one intended difference

| | Notebook 02 | Notebook 03 |
|---|---|---|
| `supervise_all` | `True` | `False` |
| Loss computed on | every one of the 32 block positions, at every denoising sub-step | only the still-masked positions at each sub-step |

The masked-only branch normalizes by the exact count of supervised positions summed over sub-steps (`n_sub·bs − per·n_sub·(n_sub−1)/2`), so per-position loss scale — and therefore the effective learning rate at 1e-5 — matches across the two arms.

### Two disclosed confounds

Beyond the objective, notebook 03 also differs in:
1. **Data volume and reuse.** 1500 distinct problems in three disjoint 500-problem pools, versus one 500-problem pool reused across all three stages.
2. **Target cleanliness.** `<think>...</think>` traces stripped; shortest surviving `\boxed{}` solution selected.

The paper reports the resulting gap as an **upper bound** on the objective's own contribution rather than a clean single-variable ablation. A perfectly matched ablation (same data, same targets, objective as the only difference) is the obvious next experiment and was not run.

### Truncation

`max_target_tokens=384` truncates longer solutions, and truncation removes the appended EOS token. Median DeepMath solution length after `<think>` stripping is well above this, so most training targets in **both** arms lack a stop token. This plausibly contributes to the non-termination symptom seen in every fine-tuned checkpoint. Notebook 03 prints the truncated fraction and offers `cfg.require_complete=True` to filter to solutions that fit and terminate.

## OPSD-1, privileged teacher (notebook 04)

| Hyperparameter | Value |
|---|---|
| Starting weights | **raw pre-SFT base** (not a stage-C checkpoint) |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Learning rate | 1e-5 |
| Gradient accumulation | 4 |
| Student schedule | 2 tokens/step (16 sub-steps per block) |
| Teacher schedule | 1 token/step (32 sub-steps per block) |
| Teacher advantage | reference solution embedded in the teacher prompt, plus the finer schedule |
| Screening temperature | **0.3** |
| Generation budget | **1536** |
| Acceptance gate | rollout contains a `\boxed{}` answer |
| Divergence | clipped KL, teacher ‖ student |
| Teacher batch | 8 sub-step states per batched forward |
| Teacher reveal order | student's own reveal order (required for batching; see below) |

**Teacher batching note.** Once the student commits a block, every teacher sub-step input state is known in advance, so they run as batched forwards rather than sequentially. This requires fixing the teacher's reveal order to the student's rather than re-ranking by the teacher's own confidence at each sub-step. The set of tokens conditioned on at each state is unchanged; only the ordering differs. Set `cfg.teacher_order="teacher_conf"` to restore the original sequential path for an A/B.

## OPSD-2, self-future teacher (notebook 05)

| Hyperparameter | Value |
|---|---|
| Student schedule | 8 tokens/step |
| Teacher schedule | 8 tokens/step (**matched** — this is the defining property) |
| Teacher prompt | identical to the student's (no reference solution) |
| Teacher advantage | `future_reveal_k = 4` positions revealed from later in the student's reveal order |
| Everything else | as OPSD-1 |

Set `future_reveal_k = 0` as a null control: the teacher becomes bit-identical to the student and the loss must be ≈0.
