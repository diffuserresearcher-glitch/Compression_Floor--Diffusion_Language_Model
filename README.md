# The Compression Floor

Code, data and results for **"The Compression Floor: Few-Step Decoding Collapse and the Preconditions for Self-Distillation in Small Block-Diffusion Language Models."**

We measure what parallel decoding costs in a small block-diffusion language model (SDAR-4B-Chat, block sizes 16 and 32) on mathematical reasoning, and test whether post-training can recover it.

**Three findings:**

1. **Few-step failure is a per-sample phase transition, not a gradual decline.** Pooled over both block sizes and all schedules (400 generations), 55% of outputs stay fluent (repeat-4 < 0.2) and 28% lock into repetition loops (> 0.7), with only 9% in between. Cell means degrade because the *collapsed fraction grows*, not because individual outputs get gradually worse.

2. **Supervision density, not data, decides whether diffusion SFT preserves or destroys the model.** On 100 problems with identical training data, masked-only supervision reaches 24.0% pooled accuracy at 1 token/step versus 11.3% for dense supervision (p = 0.00005). Changing data volume and target cleanliness has no detectable effect (p = 0.80). Masked-only SFT at the coarsest curriculum stage is statistically indistinguishable from the base model (33% vs 41%, p = 0.24).

3. **Both preconditions of on-policy self-distillation fail at this scale, and we measure why.** Trajectory yield is set by sampling, not capability (0/28 usable rollouts at temperature 0.9; 38/50 at temperature 0.3 with a larger budget). Once yield was fixed, the distillation loss collapsed. An Overlap Top-K measurement over 6,224 denoising states explains it: a reference-prefix teacher agrees with the student 85.6% of the time versus 56.7% for a self-future lookahead teacher, so the privileged construction has almost nothing to transfer.

---

## Quick start (no GPU, under a minute)

Every number and figure in the paper regenerates from the shipped results without touching a GPU or downloading a model.

```bash
git clone https://github.com/<user>/dlm-compression-floor.git
cd dlm-compression-floor
pip install -r requirements.txt          # matplotlib + numpy only

python scripts/verify_results.py         # 30 assertions against the paper's claims
python scripts/make_figures.py           # regenerates all 4 figures into results/figures/
```

`verify_results.py` is the fastest way to confirm a clone is intact. It re-derives the bimodality split from raw per-problem data, checks the collapse cliff sits between t=4 and t=8 at both block sizes, confirms the supervision-density effect is significant and the data effect is not, and validates the Overlap Top-K intervals do not overlap. If any result file is edited in a way that contradicts a paper claim, it fails loudly and names the claim.

---

## Repository layout

```
notebooks/     9 notebooks: 5 experiment pipelines + analysis/benchmark (run in order)
data/          the fixed 40-problem benchmark used by notebooks 01-07
results/       every measured artifact (CSV/JSON) + the 4 paper figures
scripts/       verify_results.py, make_figures.py  (both CPU-only)
docs/          reproduction, hyperparameters, evaluators, compute budgets
paper/         LaTeX source and bibliography
```

---

## Which file produced which paper claim

| Paper element | Source file | Produced by |
|---|---|---|
| Table 1 (base model, both block sizes) | `results/floor_grid_aggregate.csv` | notebook 01 |
| Figure 1 (bimodality, n=400) | `results/floor_grid_per_problem.csv` | notebook 01 |
| Table 2 + Figure 2 (n=100, all arms) | `results/benchmark_100_all_arms.json` | notebooks 08a + 08b |
| Figure 3 (repeat-4 degeneration) | `results/benchmark_100_all_arms.json` | notebooks 08a + 08b |
| Table 3 (trajectory yield) | `results/opsd_precondition_results.json` | notebook 04 |
| Figure 4 (Overlap Top-K) | `results/overlap_topk_results.json` | notebook 06 |
| Superseded 40-problem curriculum results | `results/sft_curriculum_benchmark.json` | notebooks 02 + 03 |

---

## Reproducing the experiments

Full detail in **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)**. Summary:

| # | Notebook | Platform | Runtime | Produces |
|---|---|---|---|---|
| 01 | `01_compression_floor_grid` | Colab A100 | ~52 min | base-model sweep, the 40-problem benchmark |
| 02 | `02_sft_dense_curriculum` | Colab A100 | ~10.3 h + 2.5 h eval | dense SFT adapters (500 problems reused) |
| 03 | `03_sft_masked_only_curriculum` | Colab A100 | ~10.3 h + 2.5 h eval | masked-only SFT adapters (1500 problems) |
| 04 | `04_opsd1_privileged_teacher` | Colab A100 | variable, run the probe first | OPSD-1 screening + loss-collapse result |
| 05 | `05_opsd2_dopsd_lookahead` | **local Jupyter** | not run to completion | OPSD-2 self-future construction |
| 06 | `06_teacher_divergence_and_cis` | Colab (§1-11) + any CPU (§12) | ~2.6 h | Overlap Top-K, bootstrap CIs |
| 07 | `07_sft_dense_matched_data` | Colab A100 | ~10.3 h + 2.5 h eval | dense SFT on data matched to notebook 03 |
| 08a | `08a_benchmark_100_raw_and_masked` | Colab A100 | ~15.5 h | n=100 results, raw + masked arms |
| 08b | `08b_benchmark_100_dense_both` | Colab A100 | ~23.3 h | n=100 results, both dense arms |

**Total: roughly 90 GPU-hours** to reproduce everything from scratch. Every notebook checkpoints and resumes, so a Colab disconnect never costs more than the item in flight.

### Order matters

Notebooks consume earlier notebooks' outputs. Running out of order fails with a missing-file assertion rather than silently producing wrong numbers. 08a and 08b are the exception: they are designed to run **in parallel on two different GPUs** and coordinate through a shared benchmark cache, verified by a hash that both must agree on.

---

## Environment constraints that will bite you

These are failure modes we actually hit. Each is handled inside the notebooks, but you need to know they exist.

**1. transformers version and import order.** `modeling_sdar.py` (shipped with the SDAR checkpoints via `trust_remote_code=True`) requires a transformers 4.5x release. Each notebook's setup cell removes `torchao` **before** any transformers import, then pins the version. Importing transformers first leaves a stale module in memory and the pin silently fails. **If the setup cell changes the installed version, restart the runtime and re-run from the top** — the notebooks print an explicit instruction when this happens.

**2. `flash_attn` is not required but is imported.** `modeling_sdar.py` imports it unconditionally. Each notebook installs a pure-PyTorch meta-path shim so the import succeeds without a compiled flash-attention build.

**3. The block-causal mask does not exist by default.** `modeling_sdar.py` ships with `_update_causal_mask` **commented out**, so whatever `attention_mask` you pass flows unmodified into attention. Without intervention the model applies *no* causal structure and is not doing block diffusion at all. Every notebook builds the mask itself: bidirectional within each block-sized grid cell, causal across cells. A sanity probe asserts (a) perturbing a token in a later cell does not change logits in an earlier cell, and (b) it *does* change earlier positions in the same cell. **Do not proceed past a failing probe.**

**4. `fuse_cross_entropy` returns `logits=None`.** In training mode with this flag enabled the model returns no logits. All notebooks force it off and never pass `labels`; loss is computed manually.

**5. Per-sub-step backward.** Mathematically identical to once per block, but keeps peak memory at one forward graph instead of `block_size / tokens_per_step` of them. This is the fix for the 1-token/step OOM.

**6. The block-4 checkpoint fails to load.** `JetLM/SDAR-4B-Chat` (block 4) does not ship `fused_linear_diffusion_cross_entropy.py`. Notebook 01 records a skip and continues. Expected, not a bug in this code.

---

## Two evaluators, never mixed

This project used two evaluation harnesses with different stopping rules and match criteria. **Their numbers are not interchangeable and are never combined in a single table or figure.** Read **[docs/EVALUATORS.md](docs/EVALUATORS.md)** before comparing anything across result files.

| | Harness A | Harness B |
|---|---|---|
| Used by | notebook 01 | notebooks 02, 03, 07, 08a, 08b |
| Budget | 1536 | 1024 |
| Stopping | repetition-collapse early stop, EOS, budget | EOS, budget |
| Match | strict | strict + lenient (numeric equivalence) |
| Base model, block 32, t=1 | **0.375** | **0.41** (on the n=100 set) |

Both numbers are correct for their own harness. The paper labels every table with which one produced it.

---

## Benchmark sets: an important caveat

There are **two** benchmark sets in this project, and they are not nested.

- **The 40-problem set** (`data/benchmark_problems.json`, 20 easy / 10 medium / 10 hard) is used by notebooks 01–07. It is frozen and shipped here so those results are exactly reproducible.
- **The 100-problem set** used by notebooks 08a/08b was drawn **independently**, not as a superset of the 40. The `in_original_40` flag came back empty for every row, meaning the notebook built a fresh set rather than extending the old one.

Consequence: the base model scores **48%** at t=1 on the 40-problem set and **41%** on the 100-problem set. That difference is sampling, not a regression. **Report n=100 as the primary benchmark and do not place n=40 numbers beside it in the same table.** The n=40 curriculum results are retained in `results/sft_curriculum_benchmark.json` for provenance only.

---

## Key hyperparameters

Full table in **[docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)**. The SFT arms (notebooks 02, 03, 07) are **identical** on everything below, which is what makes the supervision-density comparison a controlled ablation:

| | Value |
|---|---|
| Base model | `JetLM/SDAR-4B-Chat-b32`, block size 32 |
| LoRA | r=32, alpha=64, dropout=0.05, on all attention + MLP projections |
| Optimizer | AdamW, lr 1e-5, cosine schedule, 10 warmup steps |
| Batch | 1 sequence/step, gradient accumulation 4 (effective batch 4) |
| Epochs | 2 per stage, 3 stages at 4 → 2 → 1 tokens/step |
| Max target tokens | 384 |
| GPUs | 1 (no DDP, no model sharding) |

The single intended difference: `supervise_all=True` (dense: loss on all 32 block positions at every denoising sub-step) versus `False` (masked-only: loss on still-masked positions only). Loss is normalized by the exact count of supervised positions in both cases, so the effective learning rate matches.

---

## Known limitations

Stated in the paper and repeated here so nobody is surprised:

- **Single seed throughout.** At n=100, per-cell accuracy carries roughly ±8 to ±10 points of binomial error. Individual cells are large-effect observations; the pooled, significance-tested comparisons carry the claims.
- **Per-stage effects are not all resolved.** Supervision density is significant pooled and at most individual stages, but not every one.
- **All fine-tuning is LoRA on one 4B model.** Full fine-tuning and larger models are untested.
- **Most training targets are truncated.** `max_target_tokens=384` cuts longer solutions and removes the appended EOS, so most samples in **both** SFT arms train on targets that never terminate. This plausibly contributes to the non-termination symptom seen in every fine-tuned checkpoint. Notebook 03 offers `cfg.require_complete=True` to filter to solutions that fit.
- **The distillation result rests on 50 screened problems**, capped by compute (see [docs/COMPUTE.md](docs/COMPUTE.md)). It is a strong signal on a small sample, not a converged training curve.
- **Notebook 05 (OPSD-2) has not been run to completion at scale.** It is included because the Overlap Top-K evidence points toward that construction, not because we have results from it.
- **The n=100 numbers in `results/` are transcribed from run output.** The per-problem records (`rows_*.jsonl`) live on the authors' Drive; copy them into `results/` to make the bootstrap intervals independently recomputable.

---

## Data and model licenses

This repository redistributes neither models nor training data. Both are downloaded at runtime by the notebooks.

- **DeepMath-103K** (`zwhe99/DeepMath-103K`) — He et al., 2025, arXiv:2504.11456
- **SDAR checkpoints** (`JetLM/SDAR-4B-Chat-b16`, `-b32`) — Cheng et al., 2025, arXiv:2510.06303
- **TraDo** (yield replication only) — Wang et al., 2025, arXiv:2509.06949

The 40-problem benchmark in `data/` contains problem statements and reference solutions derived from DeepMath-103K; see [data/README.md](data/README.md) and respect the upstream license.

LoRA adapters produced by the notebooks are not committed (see `.gitignore`); they are large and fully reproducible.

## Citation

See [CITATION.cff](CITATION.cff). **Note:** the author block is anonymized pending publication and must be filled in before public release.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)).
