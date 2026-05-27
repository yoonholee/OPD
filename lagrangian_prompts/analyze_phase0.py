"""
Pull Phase 0 results from the Modal volume, print a summary, render Pareto plots.

Usage:
    python analyze_phase0.py                  # summarize from local cache
    python analyze_phase0.py --list           # list all result files on volume
    python analyze_phase0.py --pull           # download all to results/phase0/
    python analyze_phase0.py --plot           # make Pareto plots (requires matplotlib)

Reports both raw and length-filtered KL spreads. A response shorter than --ylen-floor
tokens is treated as length-confounded (single-token-decision dominates the per-token
KL average) and is shown separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

RESULTS_VOL_NAME = "phase0-results"
LOCAL_DIR = Path(__file__).parent / "results" / "phase0"

YLEN_FLOOR_DEFAULT = 200  # below this, KL is dominated by 1-2 decision points
KL_CAP_DEFAULT = 0.5  # drop prompts above this KL/tok as outliers


def list_remote() -> list[str]:
    vol = modal.Volume.from_name(RESULTS_VOL_NAME, create_if_missing=False)
    return sorted(e.path for e in vol.iterdir("/") if e.path.endswith(".json"))


def pull_all() -> list[Path]:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(RESULTS_VOL_NAME, create_if_missing=False)
    pulled = []
    for entry in vol.iterdir("/"):
        if not entry.path.endswith(".json"):
            continue
        local = LOCAL_DIR / Path(entry.path).name
        with open(local, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        pulled.append(local)
    return pulled


def load_local() -> list[dict]:
    if not LOCAL_DIR.exists():
        return []
    out = []
    for p in sorted(LOCAL_DIR.glob("phase0__*.json")):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception as e:
            print(f"  WARN: skipping {p.name}: {e}", file=sys.stderr)
    return out


def latest_per_combo(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Keep only the latest run per (model_id, task) pair.

    Uses the embedded epoch timestamp in the filename (saved at end of run_one).
    Falls back to wall_time_s if not parseable.
    """
    # Each record has a 'saved to' path in the file containing the timestamp; we
    # instead trust the order of glob (sorted) and pick the *last* observed per combo.
    by_combo: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["model_id"], r["task"])
        by_combo[key] = r  # overwrites; later wins
    return by_combo


def summarize(record: dict, ylen_floor: int = YLEN_FLOOR_DEFAULT) -> str:
    out = []
    out.append(f"\n=== {record['model_id']} / {record['task']} ===")
    out.append(
        f"n_problems={record['n_problems']} n_prompts={record['n_prompts']} "
        f"k={record['k']} max_tokens={record['max_tokens']}"
    )
    by_prompt = record["by_prompt"]
    sorted_prompts = sorted(by_prompt.items(), key=lambda kv: -kv[1]["acc_mean"])

    out.append(
        f"{'prompt':<22} {'acc':>6} {'±se':>5} {'kl/tok':>8} {'±se':>5} "
        f"{'ylen':>5} {'tag':>10}"
    )
    long_prompts = []
    short_prompts = []
    for pname, agg in sorted_prompts:
        ylen = agg["ylen_mean"]
        tag = "short!" if ylen < ylen_floor else ""
        if ylen < ylen_floor:
            short_prompts.append(agg)
        else:
            long_prompts.append(agg)
        out.append(
            f"{pname:<22} {agg['acc_mean']:>6.3f} {agg['acc_se']:>5.3f} "
            f"{agg['kl_mean']:>8.3f} {agg.get('kl_se', 0):>5.3f} "
            f"{ylen:>5.0f} {tag:>10}"
        )

    # All-prompts stats
    accs_all = [a["acc_mean"] for a in by_prompt.values()]
    kls_all = [a["kl_mean"] for a in by_prompt.values()]
    out.append(
        f"\n[all prompts]    acc range: {max(accs_all) - min(accs_all):.3f}  "
        f"kl range: {max(kls_all) - min(kls_all):.3f} nats/tok"
    )

    # Length-filtered stats
    if long_prompts:
        accs_l = [a["acc_mean"] for a in long_prompts]
        kls_l = [a["kl_mean"] for a in long_prompts]
        kl_range_l = max(kls_l) - min(kls_l)
        gate = "PASS" if kl_range_l >= 0.3 else "FAIL"
        out.append(
            f"[ylen>={ylen_floor}]   acc range: {max(accs_l) - min(accs_l):.3f}  "
            f"kl range: {kl_range_l:.3f} nats/tok  council gate: {gate}"
        )
        out.append(f"[ylen>={ylen_floor}]   n_kept: {len(long_prompts)}/{len(by_prompt)}")
    if short_prompts:
        out.append(
            f"[ylen<{ylen_floor}]    {len(short_prompts)} prompts excluded as length-confounded"
        )

    return "\n".join(out)


def plot_pareto(
    records: list[dict],
    out_path: Path,
    ylen_floor: int = YLEN_FLOOR_DEFAULT,
    kl_cap: float = KL_CAP_DEFAULT,
) -> None:
    import math

    import matplotlib.pyplot as plt

    n = len(records)
    nrows = 2
    ncols = math.ceil(n / nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False,
        sharex="row", sharey="row",
    )
    flat_axes = axes.flatten()
    # Hide axes for any leftover empty cells.
    for ax in flat_axes[n:]:
        ax.set_visible(False)
    for ax_i, rec in enumerate(records):
        ax = flat_axes[ax_i]
        by_prompt = rec["by_prompt"]
        # Drop length-confounded prompts and high-KL outliers.
        kept = [
            (name, agg) for name, agg in by_prompt.items()
            if agg["ylen_mean"] >= ylen_floor and agg["kl_mean"] <= kl_cap
        ]
        n_dropped = len(by_prompt) - len(kept)
        if not kept:
            ax.set_title(f"{rec['model_id'].split('/')[-1]} / {rec['task']}\n(all prompts dropped)")
            continue

        names = [k[0] for k in kept]
        accs = [k[1]["acc_mean"] for k in kept]
        kls = [k[1]["kl_mean"] for k in kept]
        acc_se = [k[1]["acc_se"] for k in kept]

        ax.errorbar(
            kls, accs, yerr=acc_se,
            fmt="o", capsize=3, ms=7, alpha=0.9, color="C0",
            markeredgecolor="black", markeredgewidth=0.8,
        )

        # Compute and draw Pareto frontier (min KL, max acc).
        sorted_pts = sorted(zip(kls, accs, names))
        frontier_x, frontier_y = [], []
        best_acc = -float("inf")
        for x, y, _ in sorted_pts:
            if y > best_acc:
                frontier_x.append(x)
                frontier_y.append(y)
                best_acc = y
        if len(frontier_x) >= 2:
            ax.plot(
                frontier_x, frontier_y,
                linestyle="--", color="black", linewidth=1.0, alpha=0.6,
                zorder=1,
            )

        for name, x, y in zip(names, kls, accs):
            ax.annotate(
                name, (x, y), fontsize=6.5, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )

        if "empty" in dict(kept):
            empty_idx = names.index("empty")
            ax.scatter(
                [kls[empty_idx]], [accs[empty_idx]],
                marker="*", s=200, color="gold", edgecolor="k", zorder=5,
                label="empty (baseline)",
            )
            ax.legend(loc="best", fontsize=8)

        task_label = {"math": "MATH500", "eqtheories": "EqTheories"}.get(
            rec["task"], rec["task"]
        )
        title = f"{rec['model_id'].split('/')[-1]} + {task_label}"
        ax.set_xlabel("KL̄ (nats/tok), forward KL  T -> θ")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ylen-floor", type=int, default=YLEN_FLOOR_DEFAULT)
    ap.add_argument("--kl-cap", type=float, default=KL_CAP_DEFAULT)
    args = ap.parse_args()

    if args.list:
        for p in list_remote():
            print(p)
        return

    if args.pull:
        pulled = pull_all()
        print(f"Pulled {len(pulled)} files to {LOCAL_DIR}")

    records = load_local()
    if not records:
        print(f"No local results in {LOCAL_DIR}. Run --pull first.", file=sys.stderr)
        return

    latest = latest_per_combo(records)
    for combo, rec in sorted(latest.items()):
        print(summarize(rec, ylen_floor=args.ylen_floor))

    if args.plot:
        out_path = LOCAL_DIR / "pareto.png"

        def _size_key(model_id: str) -> float:
            import re
            m = re.search(r"-(\d+(?:\.\d+)?)B", model_id)
            return float(m.group(1)) if m else 0.0

        # Lay out as rows=task (math top), cols=model-size (ascending).
        _task_order = {"math": 0, "eqtheories": 1}
        ordered = sorted(
            latest.values(),
            key=lambda r: (_task_order.get(r["task"], 99), _size_key(r["model_id"])),
        )
        plot_pareto(ordered, out_path, ylen_floor=args.ylen_floor, kl_cap=args.kl_cap)


if __name__ == "__main__":
    main()
