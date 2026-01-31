"""Re-generate VRR plots from saved trajectory logs.

Why:
- Some older runs produced incorrect or inconsistent VRR plots in
  `outputs/.../figures/` (e.g., plotting only a single curve or using the wrong
  per-step values).
- The saved `trajectory.json` files contain the authoritative per-step
  `variance_sum` values. This script recomputes:
  - raw VRR: (v0 - v_t) / v0
  - best-so-far VRR: (v0 - min(v_0..v_t)) / v0
  and then re-plots aggregated (mean) curves per policy.

Example:
  python3 src/scripts/replot_vrr_from_logs.py \
    --run_root outputs/batch_results/run_20260128_190351 \
    --suffix _corrected
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_gray_image(path: Path) -> np.ndarray:
    """Read an image as float32 grayscale in [0,255].

    Uses PIL to avoid adding hard deps.
    """
    from PIL import Image

    img = Image.open(path)
    img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def _semantic_visibility_from_saved_mask(traj_dir: Path, step: int) -> float:
    """Proxy semantic visibility from saved cloud render mask.

    Older runs may not store `metrics.semantic_visibility`. However, many runs
    saved debug masks:
      step_XX_cloud0_view0_mask.png
    which is (mask * 255) from CLIPSeg on the point-cloud render.

    We treat visibility as fraction of pixels with value > 127.
    """
    p = traj_dir / f"step_{int(step):02d}_cloud0_view0_mask.png"
    if not p.exists():
        return float("nan")
    g = _read_gray_image(p)
    if g.size == 0:
        return float("nan")
    return float(np.mean(g > 127.0))


def _policy_from_traj_name(traj_name: str) -> str:
    # Typical names: active_t0, active_combined_t2, geometric_t1, random_t0
    name = str(traj_name)
    if "_t" in name:
        return name.split("_t")[0]
    return name


def _compute_vrr(variance_sums: List[float]) -> List[float]:
    if not variance_sums:
        return []
    v0 = float(variance_sums[0])
    denom = v0 if v0 > 1e-8 else 1e-8
    return [(v0 - float(v)) / denom for v in variance_sums]


def _compute_vrr_best_so_far(variance_sums: List[float]) -> List[float]:
    if not variance_sums:
        return []
    v0 = float(variance_sums[0])
    denom = v0 if v0 > 1e-8 else 1e-8
    best = float("inf")
    out: List[float] = []
    for v in variance_sums:
        best = min(best, float(v))
        out.append((v0 - best) / denom)
    return out


def _compute_norm_uncertainty(variance_sums: List[float]) -> List[float]:
    """Normalized uncertainty curve u_t/u0 (lower is better)."""
    if not variance_sums:
        return []
    u0 = float(variance_sums[0])
    denom = u0 if u0 > 1e-8 else 1e-8
    return [float(u) / denom for u in variance_sums]


def _compute_norm_uncertainty_best_so_far(variance_sums: List[float]) -> List[float]:
    """Monotone normalized uncertainty curve min(u_0..u_t)/u0 (lower is better)."""
    if not variance_sums:
        return []
    u0 = float(variance_sums[0])
    denom = u0 if u0 > 1e-8 else 1e-8
    best = float("inf")
    out: List[float] = []
    for u in variance_sums:
        best = min(best, float(u))
        out.append(best / denom)
    return out


def _auc(curve: List[float]) -> float:
    """Simple AUC over steps (lower is better for uncertainty curves)."""
    if not curve:
        return float("nan")
    # Uniform step spacing; trapezoidal rule.
    y = np.asarray(curve, dtype=np.float32)
    if y.size == 1:
        return float(y[0])
    return float(np.trapz(y, dx=1.0))


def _mean_curve(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []
    min_len = min(len(c) for c in curves if c)
    if min_len <= 0:
        return []
    mat = np.asarray([c[:min_len] for c in curves], dtype=np.float32)
    return np.mean(mat, axis=0).tolist()


def _plot(curves: Dict[str, List[float]], *, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for name, curve in curves.items():
        if not curve:
            continue
        plt.plot(range(len(curve)), curve, marker="o", label=name, linewidth=2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("VRR")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _write_csv(curves: Dict[str, List[float]], path: Path) -> None:
    # Column layout: step, <policy1>, <policy2>, ... (empty if missing)
    if not curves:
        return
    names = list(curves.keys())
    max_len = max(len(v) for v in curves.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("step," + ",".join(names) + "\n")
        for i in range(max_len):
            row = [str(i)]
            for n in names:
                v = curves[n]
                row.append("" if i >= len(v) else str(float(v[i])))
            f.write(",".join(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True, help="Run folder containing per-trajectory subfolders.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <run_root>/figures).",
    )
    ap.add_argument("--suffix", type=str, default="_corrected", help="Filename suffix for regenerated plots.")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else (run_root / "figures")
    suffix = str(args.suffix)

    # Collect curves per policy.
    vrr_by_policy: Dict[str, List[List[float]]] = defaultdict(list)
    vrrb_by_policy: Dict[str, List[List[float]]] = defaultdict(list)
    unc_by_policy: Dict[str, List[List[float]]] = defaultdict(list)
    unc_best_by_policy: Dict[str, List[List[float]]] = defaultdict(list)

    # Semantic visibility curves (higher is better).
    sem_by_policy: Dict[str, List[List[float]]] = defaultdict(list)
    sem_best_by_policy: Dict[str, List[List[float]]] = defaultdict(list)

    for tj in run_root.glob("**/trajectory.json"):
        # Skip nested figures directories.
        if "figures" in tj.parts:
            continue
        data = _load_json(tj)
        cfg = data.get("config") or {}
        traj_name = (cfg.get("experiment") or {}).get("trajectory_name") or tj.parent.name
        policy = _policy_from_traj_name(str(traj_name))

        metrics = data.get("metrics") or {}
        variance_sums = metrics.get("variance_sum")
        if not isinstance(variance_sums, list) or not variance_sums:
            steps = data.get("steps") or []
            variance_sums = [float(s.get("variance_sum")) for s in steps if s.get("variance_sum") is not None]
        variance_sums = [float(v) for v in (variance_sums or [])]
        if not variance_sums:
            continue

        # Semantic visibility:
        # Prefer the value stored in metrics if available (newer runs).
        # Otherwise, fall back to reading saved debug masks if present.
        sem_vis = metrics.get("semantic_visibility")
        if isinstance(sem_vis, list) and sem_vis:
            sem_curve = [float(x) if x is not None else float("nan") for x in sem_vis]
        else:
            sem_curve = [
                _semantic_visibility_from_saved_mask(tj.parent, step=i) for i in range(len(variance_sums))
            ]

        # Best-so-far semantic visibility (monotone increasing).
        sem_best: List[float] = []
        best = -float("inf")
        for v in sem_curve:
            if not np.isfinite(v):
                sem_best.append(float("nan"))
                continue
            best = max(best, float(v))
            sem_best.append(float(best))

        vrr = metrics.get("variance_reduction_rate")
        if not isinstance(vrr, list) or len(vrr) != len(variance_sums):
            vrr = _compute_vrr(variance_sums)

        vrrb = metrics.get("variance_reduction_rate_best_so_far")
        if not isinstance(vrrb, list) or len(vrrb) != len(variance_sums):
            vrrb = _compute_vrr_best_so_far(variance_sums)

        vrr_by_policy[policy].append([float(x) for x in vrr])
        vrrb_by_policy[policy].append([float(x) for x in vrrb])

        # "Uncertainty reduction" style metrics (often easier to interpret):
        # - normalized remaining uncertainty u_t/u0 (lower is better)
        # - best-so-far remaining uncertainty min(u_0..u_t)/u0 (monotone)
        unc = _compute_norm_uncertainty(variance_sums)
        unc_best = _compute_norm_uncertainty_best_so_far(variance_sums)
        unc_by_policy[policy].append([float(x) for x in unc])
        unc_best_by_policy[policy].append([float(x) for x in unc_best])

        # Keep semantic curves only if we have at least one finite value.
        if any(np.isfinite(x) for x in sem_curve):
            sem_by_policy[policy].append([float(x) for x in sem_curve])
        if any(np.isfinite(x) for x in sem_best):
            sem_best_by_policy[policy].append([float(x) for x in sem_best])

    # Mean curves.
    avg_vrr = {p: _mean_curve(curves) for p, curves in sorted(vrr_by_policy.items())}
    avg_vrrb = {p: _mean_curve(curves) for p, curves in sorted(vrrb_by_policy.items())}
    avg_unc = {p: _mean_curve(curves) for p, curves in sorted(unc_by_policy.items())}
    avg_unc_best = {p: _mean_curve(curves) for p, curves in sorted(unc_best_by_policy.items())}

    avg_sem = {p: _mean_curve(curves) for p, curves in sorted(sem_by_policy.items())}
    avg_sem_best = {p: _mean_curve(curves) for p, curves in sorted(sem_best_by_policy.items())}

    _plot(
        avg_vrr,
        title="Variance Reduction Rate (recomputed from variance_sum)",
        path=out_dir / f"comparison_vrr{suffix}.png",
    )
    _plot(
        avg_vrrb,
        title="Variance Reduction Rate (best-so-far; recomputed from variance_sum)",
        path=out_dir / f"comparison_vrr_best_so_far{suffix}.png",
    )
    _write_csv(avg_vrr, out_dir / f"comparison_vrr{suffix}.csv")
    _write_csv(avg_vrrb, out_dir / f"comparison_vrr_best_so_far{suffix}.csv")

    # Uncertainty curves (lower is better).
    _plot(
        avg_unc,
        title="Normalized remaining uncertainty (u_t/u0; lower is better)",
        path=out_dir / f"comparison_uncertainty_norm{suffix}.png",
    )
    _plot(
        avg_unc_best,
        title="Normalized remaining uncertainty (best-so-far; lower is better)",
        path=out_dir / f"comparison_uncertainty_best_so_far_norm{suffix}.png",
    )
    _write_csv(avg_unc, out_dir / f"comparison_uncertainty_norm{suffix}.csv")
    _write_csv(avg_unc_best, out_dir / f"comparison_uncertainty_best_so_far_norm{suffix}.csv")

    # AUC summary (single-number metric; lower is better for uncertainty curves).
    auc_rows: List[Tuple[str, float, float]] = []
    for p in sorted(avg_unc.keys()):
        auc_rows.append((p, _auc(avg_unc[p]), _auc(avg_unc_best.get(p, []))))
    auc_path = out_dir / f"summary_uncertainty_auc{suffix}.csv"
    with open(auc_path, "w", encoding="utf-8") as f:
        f.write("policy,auc_uncertainty_norm,auc_uncertainty_best_so_far_norm\n")
        for p, a1, a2 in auc_rows:
            f.write(f"{p},{a1},{a2}\n")

    # Semantic plots (higher is better). These help explain cases where a policy
    # reduces *global* uncertainty but does not observe the handle.
    if avg_sem:
        _plot(
            avg_sem,
            title="Semantic visibility (proxy; higher is better)",
            path=out_dir / f"comparison_semantic_visibility{suffix}.png",
        )
        _write_csv(avg_sem, out_dir / f"comparison_semantic_visibility{suffix}.csv")

    if avg_sem_best:
        _plot(
            avg_sem_best,
            title="Semantic visibility (best-so-far; proxy; higher is better)",
            path=out_dir / f"comparison_semantic_visibility_best_so_far{suffix}.png",
        )
        _write_csv(avg_sem_best, out_dir / f"comparison_semantic_visibility_best_so_far{suffix}.csv")

    print(f"Wrote corrected plots under: {out_dir}")


if __name__ == "__main__":
    main()
