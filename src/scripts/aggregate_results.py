"""Aggregate batch experiment outputs into tables.

Reads `trajectory.json` files produced by [`ActiveHallucinationRunner.run_episode()`](src/experiments.py:118)
and emits:

- A per-run CSV (one row per mesh/policy/trial).
- A per-policy summary CSV (mean/std across runs).

This script is intentionally dependency-free (stdlib only) so it can run on VMs
without extra packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _finite(xs: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in xs:
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            out.append(xf)
    return out


def _mean(xs: Iterable[float]) -> float:
    vals = _finite(xs)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(xs: Iterable[float]) -> float:
    vals = _finite(xs)
    if len(vals) < 2:
        return float("nan")
    return float(statistics.stdev(vals))


@dataclass
class RunRow:
    run_root: str
    mesh: str
    traj: str
    policy: str
    trial: Optional[str]
    n_steps: int
    init_view: Optional[int]
    final_view: Optional[int]
    variance_0: float
    variance_final: float
    vrr_final: float
    vrr_best_final: float
    policy_score_final: float
    semantic_visibility_0: float
    semantic_visibility_1: float
    semantic_visibility_best: float
    semantic_pass_k: Optional[bool]
    distance_to_gt_final: float
    success_final: Optional[bool]


def _parse_policy_trial(traj_name: str) -> Tuple[str, Optional[str]]:
    # Convention used by run_experiments: <policy>_t<trial>
    if "_t" in traj_name:
        p, t = traj_name.rsplit("_t", 1)
        return p, t
    return traj_name, None


def load_rows(input_dir: Path, *, semantic_pass_threshold: float, semantic_pass_k: int) -> List[RunRow]:
    rows: List[RunRow] = []
    thr = float(semantic_pass_threshold)
    k = int(max(1, semantic_pass_k))
    for p in sorted(input_dir.rglob("trajectory.json")):
        # Expect: <input>/<mesh>/<traj>/trajectory.json
        if p.parent is None or p.parent.parent is None:
            continue
        traj = p.parent.name
        mesh = p.parent.parent.name
        policy, trial = _parse_policy_trial(traj)

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        steps = data.get("steps") or []
        metrics = data.get("metrics") or {}

        variance_sum = metrics.get("variance_sum") or []
        vrr = metrics.get("variance_reduction_rate") or []
        vrr_best = metrics.get("variance_reduction_rate_best_so_far") or []

        # Semantic visibility is a per-step fraction-of-pixels proxy (CLIPSeg on simulator RGB).
        # Newer runs store it in metrics; older runs may only have it per-step or not at all.
        sem_vis = metrics.get("semantic_visibility")
        if not isinstance(sem_vis, list) or not sem_vis:
            sem_vis = [s.get("semantic_visibility") for s in steps] if steps else []

        sem_best = max(_finite(sem_vis)) if sem_vis else float("nan")

        def _at(arr: Any, idx: int) -> float:
            try:
                return float(arr[idx])
            except Exception:
                return float("nan")

        sem0 = _at(sem_vis, 0)
        sem1 = _at(sem_vis, 1)

        # pass@k-style metric over steps 1..k (i.e., "after taking up to k views, did we see it?")
        sem_pass: Optional[bool] = None
        if isinstance(sem_vis, list) and len(sem_vis) > 1:
            window: List[float] = []
            for i in range(1, min(len(sem_vis), k + 1)):
                try:
                    v = float(sem_vis[i])
                except Exception:
                    continue
                if math.isfinite(v):
                    window.append(v)
            if window:
                sem_pass = bool(any(v >= thr for v in window))

        init_view = None
        final_view = None
        if steps:
            try:
                init_view = int(steps[0].get("view_id"))
                final_view = int(steps[-1].get("view_id"))
            except Exception:
                init_view = None
                final_view = None

        def _last(arr: Any) -> float:
            try:
                return float(arr[-1])
            except Exception:
                return float("nan")

        def _first(arr: Any) -> float:
            try:
                return float(arr[0])
            except Exception:
                return float("nan")

        policy_score_final = float("nan")
        if steps:
            try:
                policy_score_final = float(steps[-1].get("policy_score", float("nan")))
            except Exception:
                policy_score_final = float("nan")

        dist_final = metrics.get("distance_to_gt_at_final_step")
        try:
            dist_final_f = float(dist_final) if dist_final is not None else float("nan")
        except Exception:
            dist_final_f = float("nan")

        success_final = metrics.get("success_at_final_step")
        if success_final not in (True, False, None):
            success_final = None

        rows.append(
            RunRow(
                run_root=str(input_dir),
                mesh=mesh,
                traj=traj,
                policy=policy,
                trial=trial,
                n_steps=int(len(variance_sum)),
                init_view=init_view,
                final_view=final_view,
                variance_0=_first(variance_sum),
                variance_final=_last(variance_sum),
                vrr_final=_last(vrr),
                vrr_best_final=_last(vrr_best),
                policy_score_final=policy_score_final,
                semantic_visibility_0=sem0,
                semantic_visibility_1=sem1,
                semantic_visibility_best=float(sem_best),
                semantic_pass_k=sem_pass,
                distance_to_gt_final=dist_final_f,
                success_final=success_final,
            )
        )
    return rows


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def summarize_by_policy(rows: List[RunRow]) -> List[Dict[str, Any]]:
    by: Dict[str, List[RunRow]] = {}
    for r in rows:
        by.setdefault(r.policy, []).append(r)
    out: List[Dict[str, Any]] = []
    for pol, rr in sorted(by.items(), key=lambda kv: kv[0]):
        pass_denom = sum(1 for x in rr if x.semantic_pass_k is not None)
        pass_num = sum(1 for x in rr if x.semantic_pass_k is True)
        out.append(
            {
                "policy": pol,
                "n": len(rr),
                "mean_vrr_final": _mean(x.vrr_final for x in rr),
                "std_vrr_final": _std(x.vrr_final for x in rr),
                "mean_vrr_best_final": _mean(x.vrr_best_final for x in rr),
                "std_vrr_best_final": _std(x.vrr_best_final for x in rr),
                "mean_var_final": _mean(x.variance_final for x in rr),
                "std_var_final": _std(x.variance_final for x in rr),
                "mean_semantic_vis_1": _mean(x.semantic_visibility_1 for x in rr),
                "mean_semantic_vis_best": _mean(x.semantic_visibility_best for x in rr),
                "semantic_pass_k_rate": (pass_num / pass_denom) if pass_denom > 0 else float("nan"),
                "gt_runs": sum(1 for x in rr if math.isfinite(x.distance_to_gt_final)),
                "success_rate": (
                    sum(1 for x in rr if x.success_final is True) / max(1, sum(1 for x in rr if x.success_final is not None))
                ),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate trajectory.json outputs into tables.")
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory to scan (e.g., outputs/batch_results/run_YYYYMMDD_HHMMSS).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write summary CSVs (default: <input_dir>/figures).",
    )

    # Pass@1 proxy for "did we see the handle in the first chosen view".
    ap.add_argument(
        "--semantic_pass_threshold",
        type=float,
        default=0.01,
        help="Threshold on semantic_visibility (fraction of pixels) used for pass@k metrics.",
    )
    ap.add_argument(
        "--semantic_pass_k",
        type=int,
        default=1,
        help="k for pass@k using semantic_visibility over steps 1..k (k=1 is pass@1).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "figures")

    rows = load_rows(
        input_dir,
        semantic_pass_threshold=float(args.semantic_pass_threshold),
        semantic_pass_k=int(args.semantic_pass_k),
    )
    if not rows:
        print(f"No trajectory.json found under: {input_dir}")
        return

    per_run_path = out_dir / "summary_runs.csv"
    per_pol_path = out_dir / "summary_by_policy.csv"

    write_csv(
        per_run_path,
        header=[
            "run_root",
            "mesh",
            "traj",
            "policy",
            "trial",
            "n_steps",
            "init_view",
            "final_view",
            "variance_0",
            "variance_final",
            "vrr_final",
            "vrr_best_final",
            "policy_score_final",
            "semantic_visibility_0",
            "semantic_visibility_1",
            "semantic_visibility_best",
            "semantic_pass_k",
            "distance_to_gt_final",
            "success_final",
        ],
        rows=[
            [
                r.run_root,
                r.mesh,
                r.traj,
                r.policy,
                r.trial,
                r.n_steps,
                r.init_view,
                r.final_view,
                r.variance_0,
                r.variance_final,
                r.vrr_final,
                r.vrr_best_final,
                r.policy_score_final,
                r.semantic_visibility_0,
                r.semantic_visibility_1,
                r.semantic_visibility_best,
                r.semantic_pass_k,
                r.distance_to_gt_final,
                r.success_final,
            ]
            for r in rows
        ],
    )

    pol = summarize_by_policy(rows)
    write_csv(
        per_pol_path,
        header=[
            "policy",
            "n",
            "mean_vrr_final",
            "std_vrr_final",
            "mean_vrr_best_final",
            "std_vrr_best_final",
            "mean_var_final",
            "std_var_final",
            "mean_semantic_vis_1",
            "mean_semantic_vis_best",
            "semantic_pass_k_rate",
            "gt_runs",
            "success_rate",
        ],
        rows=[
            [
                d["policy"],
                d["n"],
                d["mean_vrr_final"],
                d["std_vrr_final"],
                d["mean_vrr_best_final"],
                d["std_vrr_best_final"],
                d["mean_var_final"],
                d["std_var_final"],
                d["mean_semantic_vis_1"],
                d["mean_semantic_vis_best"],
                d["semantic_pass_k_rate"],
                d["gt_runs"],
                d["success_rate"],
            ]
            for d in pol
        ],
    )

    print(f"Wrote per-run table: {per_run_path}")
    print(f"Wrote per-policy summary: {per_pol_path}")


if __name__ == "__main__":
    main()
