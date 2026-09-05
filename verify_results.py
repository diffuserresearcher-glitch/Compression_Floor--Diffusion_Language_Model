#!/usr/bin/env python3
"""Verify the shipped results are internally consistent, and optionally compare
a reproduction run against them.

No GPU and no model downloads required.

    python scripts/verify_results.py
    python scripts/verify_results.py --grid /path/to/your/grid_aggregate_full.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

# Values asserted in the paper. If a code change breaks one of these, the paper
# and the repository have diverged and one of them is wrong.
PAPER_CLAIMS = {
    "bimodality_fluent_frac": 0.55,   # repeat-4 < 0.2, pooled b16+b32
    "bimodality_collapsed_frac": 0.28,  # repeat-4 > 0.7
    "bimodality_middle_frac": 0.09,   # 0.3 <= repeat-4 <= 0.6
    "bimodality_n": 400,
}

failures = []
notes = []


def check(condition, message):
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        failures.append(message)


def approx(a, b, tol=0.01):
    return abs(a - b) <= tol


def verify_benchmark_set():
    print("\n[1] Benchmark problem set")
    path = DATA / "benchmark_problems.json"
    if not path.exists():
        failures.append("data/benchmark_problems.json missing")
        print("  FAIL  file missing")
        return
    problems = json.load(open(path))
    check(len(problems) == 40, f"40 problems present (found {len(problems)})")
    tiers = {}
    for p in problems:
        t = p.get("tier", "?")
        tiers[t] = tiers.get(t, 0) + 1
    check(tiers.get("easy") == 20, f"20 easy (found {tiers.get('easy')})")
    check(tiers.get("medium") == 10, f"10 medium (found {tiers.get('medium')})")
    check(tiers.get("hard") == 10, f"10 hard (found {tiers.get('hard')})")


def verify_bimodality():
    print("\n[2] Bimodality claim (paper Figure 1, Section 4.1)")
    path = RESULTS / "floor_grid_per_problem.csv"
    if not path.exists():
        failures.append("results/floor_grid_per_problem.csv missing")
        print("  FAIL  file missing")
        return
    rows = list(csv.DictReader(open(path)))
    vals = [float(r["repeat4"]) for r in rows if r["repeat4"] not in ("", "nan")]
    n = len(vals)
    lo = sum(1 for v in vals if v < 0.2) / n
    mid = sum(1 for v in vals if 0.3 <= v <= 0.6) / n
    hi = sum(1 for v in vals if v > 0.7) / n
    check(n == PAPER_CLAIMS["bimodality_n"], f"n = {PAPER_CLAIMS['bimodality_n']} generations (found {n})")
    check(approx(lo, PAPER_CLAIMS["bimodality_fluent_frac"]), f"fluent fraction ~{PAPER_CLAIMS['bimodality_fluent_frac']:.0%} (found {lo:.0%})")
    check(approx(hi, PAPER_CLAIMS["bimodality_collapsed_frac"]), f"collapsed fraction ~{PAPER_CLAIMS['bimodality_collapsed_frac']:.0%} (found {hi:.0%})")
    check(approx(mid, PAPER_CLAIMS["bimodality_middle_frac"]), f"middle band ~{PAPER_CLAIMS['bimodality_middle_frac']:.0%} (found {mid:.0%})")
    check(lo > mid and hi > mid, "distribution is bimodal (both modes exceed the middle band)")


def verify_cliff():
    print("\n[3] Collapse cliff location (paper Table 1)")
    path = RESULTS / "floor_grid_aggregate.csv"
    if not path.exists():
        failures.append("results/floor_grid_aggregate.csv missing")
        print("  FAIL  file missing")
        return
    grid = {}
    for r in csv.DictReader(open(path)):
        grid[(r["block_size"], int(r["tokens_per_step"]))] = r
    for block in ("16", "32"):
        t4 = float(grid[(block, 4)]["correct"])
        t8 = float(grid[(block, 8)]["correct"])
        check(t4 > 0 and t8 == 0.0,
              f"block {block}: cliff between t=4 ({t4:.3f}) and t=8 ({t8:.3f})")
    # block 16 uniformly at or above block 32 on correct rate
    uniform = all(float(grid[("16", t)]["correct"]) >= float(grid[("32", t)]["correct"])
                  for t in (1, 2, 4, 8, 16))
    check(uniform, "block 16 is at or above block 32 on correct rate at every schedule")


def verify_curriculum():
    print("\n[4] Supervision-density deltas (paper Table 2)")
    path = RESULTS / "sft_curriculum_benchmark.json"
    if not path.exists():
        failures.append("results/sft_curriculum_benchmark.json missing")
        print("  FAIL  file missing")
        return
    d = json.load(open(path))
    for stage in ("s1_t4", "s2_t2", "s3_t1"):
        dense = d["dense_supervise_all"][stage]["tokens_per_step_1"]["overall"]
        masked = d["masked_only"][stage]["tokens_per_step_1"]["overall"]
        recorded = d["headline_deltas_at_t1"][stage]
        check(approx(masked - dense, recorded, 0.001),
              f"{stage}: masked-only minus dense = {masked - dense:+.2f} (recorded {recorded:+.2f})")
    check(all(d["masked_only"][s]["tokens_per_step_1"]["overall"]
              > d["dense_supervise_all"][s]["tokens_per_step_1"]["overall"]
              for s in ("s1_t4", "s2_t2", "s3_t1")),
          "masked-only beats dense at every stage (t=1)")
    base = d["base"]["tokens_per_step_1"]["overall"]
    best = max(d["masked_only"][s]["tokens_per_step_1"]["overall"]
               for s in ("s1_t4", "s2_t2", "s3_t1"))
    check(best < base,
          f"no fine-tuned checkpoint restores base accuracy ({best:.2f} < {base:.2f})")


def verify_opsd():
    print("\n[5] OPSD preconditions (paper Table 3, Section 4.3)")
    path = RESULTS / "opsd_precondition_results.json"
    if not path.exists():
        failures.append("results/opsd_precondition_results.json missing")
        print("  FAIL  file missing")
        return
    d = json.load(open(path))
    runs = d["precondition_1_trajectory_yield"]["runs"]
    hot = [r for r in runs if r["temperature"] == 0.9]
    cold = [r for r in runs if r["temperature"] == 0.3]
    check(all(r["usable"] == 0 for r in hot),
          f"all temperature-0.9 runs yielded zero usable rollouts ({len(hot)} runs)")
    check(all(r["usable"] > 0 for r in cold),
          "temperature-0.3 run yielded usable rollouts")
    check(len({r["model"] for r in hot}) >= 2,
          "yield collapse replicated on at least two model families")
    check(d["precondition_2_teacher_student_divergence"]["outcome"] == "loss_collapse",
          "precondition 2 recorded as loss collapse")


def compare_reproduction(user_grid):
    print(f"\n[6] Comparing your run against ours: {user_grid}")
    ours = {}
    for r in csv.DictReader(open(RESULTS / "floor_grid_aggregate.csv")):
        ours[(r["block_size"], int(r["tokens_per_step"]))] = r
    yours = {}
    for r in csv.DictReader(open(user_grid)):
        yours[(r["block_size"], int(r["tokens_per_step"]))] = r
    shared = sorted(set(ours) & set(yours))
    if not shared:
        print("  no overlapping cells found; check the file's block_size/tokens_per_step columns")
        return
    print(f"  {'cell':<14}{'ours':>8}{'yours':>8}{'diff':>8}")
    big = 0
    for key in shared:
        o = float(ours[key]["correct"])
        y = float(yours[key]["correct"])
        flag = ""
        if abs(o - y) > 0.10:
            flag = "  <-- exceeds sampling noise"
            big += 1
        print(f"  b{key[0]} t{key[1]:<9}{o:>8.3f}{y:>8.3f}{y - o:>+8.3f}{flag}")
    if big:
        notes.append(f"{big} cell(s) differ by more than 0.10 on correct rate. "
                     "With n=40 that exceeds plausible sampling noise; check config "
                     "(gen_budget, decoding, benchmark problem set) before trusting the run.")
    else:
        print("\n  All cells within +/-0.10 of ours, consistent with n=40 sampling noise.")


def verify_benchmark100():
    print("\n[6] n=100 benchmark headline claims")
    path = RESULTS / "benchmark_100_all_arms.json"
    if not path.exists():
        failures.append("results/benchmark_100_all_arms.json missing")
        print("  FAIL  file missing"); return
    d = json.load(open(path))
    sig = d["significance_tests"]

    dens = sig["supervision_density_masked_vs_dense1500"]
    for t in ("t1", "t2"):
        pooled = dens[t]["pooled"]
        check(pooled["p"] < 0.05,
              f"{t}: supervision density significant (masked {pooled['masked']:.1%} vs "
              f"dense1500 {pooled['dense1500']:.1%}, p={pooled['p']})")
        check(pooled["masked"] > pooled["dense1500"],
              f"{t}: masked-only outperforms dense on matched data")

    data = sig["data_effect_dense1500_vs_dense500"]
    for t in ("t1", "t2"):
        pooled = data[t]["pooled"]
        check(pooled["p"] > 0.05,
              f"{t}: data effect NOT significant (p={pooled['p']}), so the gap is not "
              f"explained by data volume or cleanliness")

    raw_t = sig["vs_raw_baseline_t1"]
    check(raw_t["masked_s1"]["p"] > 0.05,
          f"masked s1 not distinguishable from raw at t=1 (p={raw_t['masked_s1']['p']})")
    check(raw_t["dense1500_s1"]["p"] < 0.05,
          f"dense1500 s1 IS significantly below raw at t=1 (p={raw_t['dense1500_s1']['p']})")

    t1 = d["tokens_per_step_1"]
    masked_r4 = max(t1[k]["repeat4"] for k in ("masked_s1","masked_s2","masked_s3"))
    dense_r4 = min(t1[k]["repeat4"] for k in ("dense1500_s1","dense1500_s2","dense1500_s3"))
    check(masked_r4 < dense_r4,
          f"repeat-4 separates the arms cleanly (masked max {masked_r4:.3f} < "
          f"dense1500 min {dense_r4:.3f})")


def verify_overlap():
    print("\n[7] Overlap Top-K teacher divergence")
    path = RESULTS / "overlap_topk_results.json"
    if not path.exists():
        failures.append("results/overlap_topk_results.json missing")
        print("  FAIL  file missing"); return
    d = json.load(open(path))
    A = d["teacher_A_privileged_prefix"]["mean_overlap"]
    B = d["teacher_B_self_future_lookahead"]["mean_overlap"]
    ciA = d["teacher_A_privileged_prefix"]["ci95"]
    ciB = d["teacher_B_self_future_lookahead"]["ci95"]
    check(A > 0.8, f"privileged teacher shows near-total agreement ({A:.3f})")
    check(B < A, f"self-future teacher diverges more from the student ({B:.3f} < {A:.3f})")
    check(ciB[1] < ciA[0],
          f"the two constructions' 95% CIs do not overlap ([{ciB[0]},{ciB[1]}] vs [{ciA[0]},{ciA[1]}])")
    check(abs((A - B) - d["gap"]["value"]) < 0.002,
          f"recorded gap {d['gap']['value']} matches the means")
    check(d["setup"]["n_states_measured"] > 5000,
          f"measured on {d['setup']['n_states_measured']} states")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", help="path to your own grid_aggregate_full.csv to compare")
    args = ap.parse_args()

    print("=" * 68)
    print("Verifying shipped results against the paper's claims")
    print("=" * 68)

    verify_benchmark_set()
    verify_bimodality()
    verify_cliff()
    verify_curriculum()
    verify_opsd()
    verify_benchmark100()
    verify_overlap()
    if args.grid:
        compare_reproduction(args.grid)

    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nThe repository and the paper have diverged. Do not cite these numbers")
        print("until the discrepancy is resolved.")
    else:
        print("All checks passed. Shipped results match the paper's claims.")
    for n in notes:
        print(f"\nNOTE: {n}")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
