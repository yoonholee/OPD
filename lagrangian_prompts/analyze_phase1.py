"""
Pull Phase 1 results, render the 4-flavor Pareto plot per (model, task) cell.

Per cell, four panels: (acc) vs (fwd_kl, rev_kl, jsd, fwd_kl_first32). Each
condition colored by family. SDR expected at top-right; anchor at bottom-left.

Usage:
    python analyze_phase1.py --pull       # download all to results/phase1/
    python analyze_phase1.py              # summarize from local cache
    python analyze_phase1.py --plot       # render per-cell Pareto plot grid
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import modal

RESULTS_VOL_NAME = "phase1-results"
LOCAL_DIR = Path(__file__).parent / "results" / "phase1"

KL_FLAVORS = ("fwd_kl", "rev_kl", "jsd", "fwd_kl_first32")
FAMILY_COLORS = {
    "anchor": "black",
    "distrib": "C0",
    "sdft": "C2",
    "sdpo": "C1",
    "sdpo_v": "C5",   # SDPO variants (no_answer, critique)
    "opsd": "C4",
    "sdr": "C3",
    "sdr_v": "C8",    # SDR variants (strategy, first_step)
}
FAMILY_MARKERS = {
    "anchor": "*",
    "distrib": "o",
    "sdft": "s",
    "sdpo": "^",
    "sdpo_v": "v",
    "opsd": "P",
    "sdr": "D",
    "sdr_v": "d",
}

# Tier ordering for grouped tables. Earlier tiers use less external information.
TIER_ORDER = [
    "none",
    "self",
    "other_solutions",
    "answer",
    "partial_solution",
    "full_solution",
]
TIER_LABEL = {
    "none": "Tier 0: problem only",
    "self": "Tier 1: + model's own draft/critique",
    "other_solutions": "Tier 2: + other problems' solutions (ICL)",
    "answer": "Tier 3: + this problem's gold answer",
    "partial_solution": "Tier 4: + slice of this problem's gold solution",
    "full_solution": "Tier 5: + full gold solution",
}


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
    for p in sorted(LOCAL_DIR.glob("phase1__*.json")):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception as e:
            print(f"  WARN: skipping {p.name}: {e}", file=sys.stderr)
    return out


def latest_per_combo(records: list[dict]) -> dict[tuple[str, str], dict]:
    by_combo: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["model_id"], r["task"])
        by_combo[key] = r  # later wins (sorted glob)
    return by_combo


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns NaN if input < 3 valid pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    n = len(pairs)
    rx = _ranks([p[0] for p in pairs])
    ry = _ranks([p[1] for p in pairs])
    mean_x = statistics.mean(rx)
    mean_y = statistics.mean(ry)
    cov = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    var_x = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    var_y = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x * var_y)


def _ranks(xs: list[float]) -> list[float]:
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j + 1 < len(sorted_idx) and xs[sorted_idx[j + 1]] == xs[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def summarize(record: dict) -> str:
    out = []
    out.append(f"\n=== {record['model_id']} / {record['task']} ===")
    out.append(
        f"n_problems={record['n_problems']} n_conditions={record['n_conditions']} "
        f"k={record['k']} max_tokens={record['max_tokens']}"
    )
    anchor = record["anchor"]
    out.append(
        f"\n  ANCHOR: acc={anchor['acc_mean']:.3f} ± {anchor['acc_se']:.3f}  "
        f"ylen={anchor['ylen_mean']:.0f}"
    )

    by_cond = record["by_condition"]
    # Group by info_tier (fall back to "none" if not tagged — older runs).
    by_tier: dict[str, list] = {}
    for cname, agg in by_cond.items():
        tier = agg.get("info_tier", "none")
        by_tier.setdefault(tier, []).append((cname, agg))

    for tier in TIER_ORDER:
        if tier not in by_tier:
            continue
        out.append(f"\n  {TIER_LABEL[tier]}")
        out.append(
            f"  {'condition':<26} {'fam':<8} {'acc':>6} "
            f"{'fwd':>7} {'rev':>7} {'jsd':>7} {'fwd32':>7} {'ylen_T':>6}"
        )
        rows = sorted(by_tier[tier], key=lambda kv: -kv[1]["acc_mean"])
        for cname, agg in rows:
            out.append(
                f"  {cname:<26} {agg['family']:<8} {agg['acc_mean']:>6.3f} "
                f"{agg['fwd_kl']:>7.3f} {agg['rev_kl']:>7.3f} "
                f"{agg['jsd']:>7.3f} {agg['fwd_kl_first32']:>7.3f} "
                f"{agg['ylen_mean_T']:>6.0f}"
            )

    # Spearman(acc, KL) on distrib-only — does the Lagrangian knob mean anything?
    distrib = [(c, a) for c, a in by_cond.items() if a["family"] == "distrib"]
    if len(distrib) >= 3:
        accs = [a["acc_mean"] for _, a in distrib]
        out.append(f"\n  Spearman(acc, KL) on distrib (n={len(distrib)}):")
        for flavor in KL_FLAVORS:
            kls = [a[flavor] for _, a in distrib]
            rho = _spearman(accs, kls)
            out.append(f"    {flavor:<18} ρ = {rho:+.3f}")

    # Family-level summary: mean (acc, KL...) per family. NaNs (typically from
    # length-filtered short rollouts) are skipped per metric so one bad condition
    # doesn't poison the family mean.
    fam_groups: dict[str, list[dict]] = {}
    for c, a in by_cond.items():
        fam_groups.setdefault(a["family"], []).append(a)
    def _nanmean(xs):
        clean = [x for x in xs if not math.isnan(x)]
        return statistics.mean(clean) if clean else float("nan")
    out.append(f"\n  Family means (NaN-skip):")
    out.append(f"    {'family':<10} {'n':>3} {'acc':>6} {'fwd':>7} {'rev':>7} {'jsd':>7} {'fwd32':>7}")
    for fam, group in fam_groups.items():
        out.append(
            f"    {fam:<10} {len(group):>3} "
            f"{_nanmean(a['acc_mean'] for a in group):>6.3f} "
            f"{_nanmean(a['fwd_kl'] for a in group):>7.3f} "
            f"{_nanmean(a['rev_kl'] for a in group):>7.3f} "
            f"{_nanmean(a['jsd'] for a in group):>7.3f} "
            f"{_nanmean(a['fwd_kl_first32'] for a in group):>7.3f}"
        )

    return "\n".join(out)


def plot_cell(record: dict, out_path: Path) -> None:
    """One row of 4 Pareto panels for this (model, task)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    by_cond = record["by_condition"]
    anchor = record["anchor"]

    for ax, flavor in zip(axes, KL_FLAVORS):
        # Plot the anchor at (0, anchor_acc).
        ax.scatter(
            [0.0], [anchor["acc_mean"]],
            marker=FAMILY_MARKERS["anchor"], s=180,
            color=FAMILY_COLORS["anchor"], edgecolor="white", linewidth=1.2,
            zorder=5, label="anchor",
        )

        # Conditions, grouped by family for legend dedup.
        seen_families = {"anchor"}
        for cname, agg in by_cond.items():
            kl = agg[flavor]
            if math.isnan(kl):
                continue
            fam = agg["family"]
            label = fam if fam not in seen_families else None
            seen_families.add(fam)
            ax.scatter(
                [kl], [agg["acc_mean"]],
                marker=FAMILY_MARKERS.get(fam, "o"), s=70,
                color=FAMILY_COLORS.get(fam, "gray"),
                edgecolor="black", linewidth=0.5, alpha=0.85,
                label=label, zorder=3,
            )
            ax.annotate(
                cname.split(":")[-1] if ":" in cname else cname,
                (kl, agg["acc_mean"]),
                fontsize=6, alpha=0.6,
                xytext=(3, 3), textcoords="offset points",
            )

        # Pareto frontier (max acc at each KL — top-left envelope).
        pts = [(0.0, anchor["acc_mean"])] + [
            (a[flavor], a["acc_mean"]) for a in by_cond.values()
            if not math.isnan(a[flavor])
        ]
        pts.sort()
        front_x, front_y = [], []
        best = -float("inf")
        for x, y in pts:
            if y > best:
                front_x.append(x)
                front_y.append(y)
                best = y
        if len(front_x) >= 2:
            ax.plot(front_x, front_y, "--", color="black", lw=0.9, alpha=0.5, zorder=1)

        ax.set_xlabel(f"{flavor} (nats/tok)")
        ax.set_title(flavor)
        ax.grid(alpha=0.3)
        if flavor == KL_FLAVORS[0]:
            ax.set_ylabel("accuracy")

    axes[0].legend(loc="best", fontsize=7)
    title = f"{record['model_id'].split('/')[-1]} / {record['task']}  (n={record['n_problems']}, K={record['k']})"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


TIER_COLORS = {
    "none": "C0",
    "self": "C5",
    "other_solutions": "C2",
    "answer": "C4",
    "partial_solution": "C8",
    "full_solution": "C3",
}


def plot_summary_grid(
    records: list[dict],
    out_path: Path,
    flavor: str = "fwd_kl_first32",
) -> None:
    """2 rows (task) × 3 cols (model size). Pareto scatter colored by info tier.

    The single figure that collapses the 6-cell story: accuracy vs KL, anchor as
    a black star at x=0, dashed top-left Pareto envelope, points by info tier.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def size_key(mid: str) -> float:
        m = re.search(r"-(\d+(?:\.\d+)?)B", mid)
        return float(m.group(1)) if m else 0.0

    by_combo = {(r["model_id"], r["task"]): r for r in records}
    tasks = ["math", "eqtheories"]
    models = sorted({m for (m, _) in by_combo}, key=size_key)

    fig, axes = plt.subplots(
        len(tasks), len(models),
        figsize=(5.0 * len(models), 4.2 * len(tasks)),
        sharey="row",
    )
    for ri, task in enumerate(tasks):
        for ci, model in enumerate(models):
            ax = axes[ri][ci]
            rec = by_combo.get((model, task))
            if rec is None:
                ax.set_visible(False)
                continue
            anchor = rec["anchor"]
            ax.scatter([0.0], [anchor["acc_mean"]], marker="*", s=240,
                       color="black", edgecolor="white", linewidth=1.3, zorder=6)
            ax.annotate("anchor", (0.0, anchor["acc_mean"]), fontsize=6.5,
                        alpha=0.7, xytext=(4, 4), textcoords="offset points")
            for cname, agg in rec["by_condition"].items():
                kl = agg.get(flavor, float("nan"))
                if math.isnan(kl):
                    continue
                tier = agg.get("info_tier", "none")
                ax.scatter([kl], [agg["acc_mean"]], s=70,
                           color=TIER_COLORS.get(tier, "gray"),
                           edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
                ax.annotate(cname.split(":")[-1] if ":" in cname else cname,
                            (kl, agg["acc_mean"]), fontsize=5.5, alpha=0.55,
                            xytext=(3, 3), textcoords="offset points")
            pts = sorted([(0.0, anchor["acc_mean"])] + [
                (a[flavor], a["acc_mean"]) for a in rec["by_condition"].values()
                if not math.isnan(a.get(flavor, float("nan")))
            ])
            fx, fy, best = [], [], -float("inf")
            for x, y in pts:
                if y > best:
                    fx.append(x); fy.append(y); best = y
            if len(fx) >= 2:
                ax.plot(fx, fy, "--", color="black", lw=1.0, alpha=0.5, zorder=1)
            ax.set_title(f"{model.split('/')[-1]} / {task}", fontsize=11)
            ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel("accuracy")
            if ri == len(tasks) - 1:
                ax.set_xlabel(f"{flavor} (nats/tok)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markeredgecolor="black", markersize=9, label=TIER_LABEL[t])
        for t, c in TIER_COLORS.items()
    ] + [Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                markeredgecolor="black", markersize=14, label="anchor (π_θ)")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    _flavor_name = {
        "fwd_kl_tk": "truncated top-k forward KL",
        "fwd_kl_first32": "first-32-token forward KL",
        "fwd_kl": "per-token forward KL (k1)",
        "fwd_kl_k3": "forward KL (k3)",
    }.get(flavor, flavor)
    fig.suptitle(
        f"Accuracy vs {_flavor_name} — Pareto-top tier shifts per cell",
        fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--plot", action="store_true")
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
        print(summarize(rec))

    if args.plot:
        for combo, rec in sorted(latest.items()):
            safe = f"{rec['model_id'].replace('/', '__')}__{rec['task']}"
            out_path = LOCAL_DIR / f"pareto_{safe}.png"
            plot_cell(rec, out_path)


if __name__ == "__main__":
    main()
