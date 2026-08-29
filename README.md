# The Compression Floor

Code and results for **"The Compression Floor: Few-Step Decoding Collapse and the Preconditions for Self-Distillation in Small Block-Diffusion Language Models."**

We measure what parallel decoding costs in a small block-diffusion language model (SDAR-4B-Chat, block sizes 16 and 32) on a fixed 40-problem mathematical reasoning benchmark, and test whether post-training can recover it. Three results:

1. **Few-step failure is a per-sample phase transition, not a gradual decline.** Pooled over schedules and both block sizes (400 generations), 55% of outputs stay fluent (repeat-4 < 0.2) and 28% lock into repetition loops (> 0.7), with only 9% in between. Cell means degrade only because the collapsed fraction grows.
2. **Supervised fine-tuning moves the collapse cliff one octave earlier**, from 4 tokens/step to 2, in every variant we trained. Supervision density is the largest factor we varied: dense supervision over all block positions drops 1-token/step accuracy from 48% to 8% across a three-stage curriculum; the standard masked-only objective holds it at 20%.
3. **Both preconditions of on-policy self-distillation fail at this scale.** Trajectory yield is set by sampling, not capability (0/28 usable rollouts at temperature 0.9; 38/50 at temperature 0.3 with a larger budget). Once yield was fixed, the privileged teacher's distribution was too close to the student's to supply gradient, and the distillation loss collapsed.

## Repository layout

```
notebooks/
  01_compression_floor_grid.ipynb        Base-model sweep: block 16 & 32 x tokens/step {1,2,4,8,16}
  02_sft_dense_curriculum.ipynb          Dense SFT (supervise_all=True), 500 problems x 3 stages
  03_sft_masked_only_curriculum.ipynb    Masked-only SFT, 1500 problems (500/stage) x 3 stages
  04_opsd1_privileged_teacher.ipynb      OPSD-1: reference-solution teacher, from raw pre-SFT weights
  05_opsd2_dopsd_lookahead.ipynb         OPSD-2: self-future/lookahead teacher, local Jupyter
data/
  benchmark_problems.json                The fixed 40 problems (20 easy / 10 medium / 10 hard)
results/                                 All measured artifacts (CSV/JSON) + figures
scripts/
  make_figures.py                        Regenerates paper figures from results/
  verify_results.py                      Checks a reproduction run against recorded numbers
docs/                                    Reproduction, hyperparameters, evaluators, compute
paper/                                   LaTeX source and bibliography
```

## Quick start

```bash
git clone https://github.com/<user>/dlm-compression-floor.git
cd dlm-compression-floor
pip install -r requirements.txt

# Regenerate the paper figures from the shipped results (no GPU needed, ~5 seconds)
python scripts/make_figures.py

# Check the shipped results are internally consistent (no GPU needed)
python scripts/verify_results.py
```

To reproduce the experiments themselves, see **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)**. Notebooks 01–04 are written for Google Colab with a Drive mount; notebook 05 is written for a local Jupyter install. Every notebook is resumable: re-running after a crash picks up from the last checkpoint.

## Reproducing the paper's numbers without a GPU

Every number in the paper is derived from files in `results/`. The figures and all reported statistics regenerate from those files alone:

| Paper element | Source file |
|---|---|
| Table 1 (base model, both block sizes) | `results/floor_grid_aggregate.csv` |
| Figure 1 (bimodality, n=400) | `results/floor_grid_per_problem.csv` |
| Figure 2 (base vs dense SFT) | `results/floor_grid_aggregate.csv` + `results/sft_dense_schedule_grid.csv` |
| Table 2 (supervision density) | `results/sft_curriculum_benchmark.json` |
| Table 3 (trajectory yield) | `results/opsd_precondition_results.json` |

## Two evaluators, never mixed

This project used two evaluation harnesses with different stopping rules. **Numbers from the two are not comparable and are never mixed within a single table or figure in the paper.** See [docs/EVALUATORS.md](docs/EVALUATORS.md) before comparing anything across result files. In short: the compression-floor grid (notebook 01) uses a repetition-collapse early stop and reports greedy `correct`; the SFT curriculum notebooks (02, 03) use a separate harness that also reports a lenient string match. The base model scores 0.375 correct at block 32 / t=1 under the first and 0.48 lenient under the second. Both are correct for their own harness.

## Known limitations

These are stated plainly in the paper and repeated here so nobody is surprised:

- **n = 40 problems, single seed.** Per-cell accuracy carries binomial error near ±15 points. The size is a compute decision, quantified in [docs/COMPUTE.md](docs/COMPUTE.md).
- **The supervision-density comparison is not a perfectly clean ablation.** The masked-only arm also used fresher data (3 disjoint pools vs 1 reused pool) and stripped reasoning traces from targets. The reported gap is an upper bound on the objective's own contribution.
- **All fine-tuning is LoRA on one 4B model.** Full fine-tuning and larger models are untested.
- **The distillation result is from 50 screened problems.** It is a strong signal on a small sample, not a converged training curve.
- **Notebook 05 (OPSD-2) has not been run to completion at full scale.** It is included because it is the construction the evidence points toward, not because we have results from it.

## Citation

If you use this code or these results, please cite the paper (see [CITATION.cff](CITATION.cff)).

## License

Code in this repository is released under the MIT License (see [LICENSE](LICENSE)). The DeepMath-103K dataset and the SDAR and TraDo model checkpoints are the property of their respective authors under their own licenses; this repository redistributes neither. See [data/README.md](data/README.md).
