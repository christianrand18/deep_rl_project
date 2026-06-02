"""Inspect committed meta-policy (obs, action, value, return) tuples.

Usage:
    python analyze_dump.py [ep_min] [picard.jsonl] [seq.jsonl]

Each dump line (from --dump_meta_path) is one committed transition:
    {"ep", "day", "obs":[7], "act", "value", "ret"}

Reports, per run, restricted to episodes >= ep_min:
  - within-episode return std  (what the critic must explain each update)
  - R^2(return ~ obs)          (how well the observation predicts the return)
  - value residual std         (return - committed value; ~sqrt(critic_loss) pre-fit)
  - per-component obs spread + single-component R^2

The 100-ep local test showed Picard==sequential early; run this at ep_min~1000+
on a long run to catch where the gap emerges and which obs component carries it.
"""
import json
import sys
import numpy as np

LABELS = ["M", "profit", "price", "price_opp", "reb_cost", "capture", "d/N"]


def load(path, ep_min):
    obs, act, val, ret, eps = [], [], [], [], []
    for line in open(path):
        d = json.loads(line)
        if d["ep"] < ep_min:
            continue
        obs.append(d["obs"]); act.append(d["act"])
        val.append(d["value"]); ret.append(d["ret"]); eps.append(d["ep"])
    return (np.array(obs), np.array(act), np.array(val),
            np.array(ret), np.array(eps))


def r2(x, y):
    """R^2 of the best linear fit y ~ [x, 1]."""
    A = np.column_stack([np.atleast_2d(x).reshape(len(y), -1), np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def analyze(path, label, ep_min):
    obs, act, val, ret, eps = load(path, ep_min)
    if len(ret) == 0:
        print(f"\n=== {label} ({path}) — no episodes >= {ep_min} ===")
        return
    n_ep = len(set(eps.tolist()))
    print(f"\n=== {label}  ({path})  episodes>={ep_min}: {n_ep}, transitions: {len(ret)} ===")
    print(f"  return: mean={ret.mean():8.3f}  std(all)={ret.std():.3f}")
    wspread = [ret[eps == e].std() for e in set(eps.tolist()) if (eps == e).sum() > 1]
    print(f"  within-episode return std (mean over eps): {np.mean(wspread):.3f}")
    print(f"  R^2(return ~ full obs, linear):  {r2(obs, ret):.3f}")
    print(f"  value residual std (return - value): {(ret - val).std():.3f}")
    print(f"  per-component obs spread + single-component R^2:")
    for i, lab in enumerate(LABELS):
        print(f"      {lab:10s} std={obs[:, i].std():7.3f}   R^2(ret~this alone)={r2(obs[:, i], ret):6.3f}")


if __name__ == "__main__":
    ep_min = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    picard = sys.argv[2] if len(sys.argv) > 2 else "saved_files/meta_dumps/picard.jsonl"
    seq = sys.argv[3] if len(sys.argv) > 3 else "saved_files/meta_dumps/seq.jsonl"
    analyze(picard, "PICARD", ep_min)
    analyze(seq, "SEQUENTIAL", ep_min)
