"""
Pull Phase 2 OPD results, render the master table + learning-curve figure.

Schema (per teacher): results/phase2/phase2__<teacher>.json =
  {"config": {...}, "history": [{step, acc_id, acc_gsm8k, acc_math_l5,
   train_loss, entropy}, ...]}  (step-0 row has train_loss/entropy = null).

Master table per teacher: base (step 0) -> final acc on each eval set, Δ, and
step-to-half-lift (first step reaching base + 0.5·(final−base)). The learning-
curve figure is 3 panels (ID / GSM8K / MATH-L5), one line per teacher.

The result is the leak-vs-transfer gap: no-leak `concise` vs leak
`aj_concise`/`sdr` vs the `anchor` KL≈0 control, on ID vs OOD.

Usage:
    python analyze_phase2.py --pull        # download to results/phase2/
    python analyze_phase2.py               # master table from local cache
    python analyze_phase2.py --plot        # + experiments/figs/004_*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_VOL_NAME = "phase2-results"
LOCAL_DIR = Path(__file__).parent / "results" / "phase2"
FIG_PATH = Path(__file__).parent / "experiments" / "figs" / "004_opd_learning_curves.png"

TEACHERS = ["anchor", "concise", "aj_concise", "sdr"]
TEACHER_TIER = {  # for the table — what the teacher sees beyond the problem
    "anchor": "— (KL≈0 control)",
    "concise": "t0 (no leak)",
    "aj_concise": "t3 (answer leak)",
    "sdr": "t5 (full solution)",
}
EVAL_SETS = [("acc_id", "MATH-500 (ID)"),
             ("acc_gsm8k", "GSM8K (OOD-down)"),
             ("acc_math_l5", "MATH-L5 (OOD-up)")]
TEACHER_COLOR = {"anchor": "black", "concise": "C0",
                 "aj_concise": "C1", "sdr": "C3"}


def pull_all() -> list[Path]:
    import modal

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


def load_local() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not LOCAL_DIR.exists():
        return out
    for p in sorted(LOCAL_DIR.glob("phase2__*.json")):
        try:
            with open(p) as f:
                rec = json.load(f)
            out[rec["config"]["teacher"]] = rec
        except Exception as e:
            print(f"  WARN skip {p.name}: {e}", file=sys.stderr)
    return out


def _half_lift_step(history: list[dict], key: str) -> int | None:
    """First step whose acc reaches base + 0.5·(final−base). None if no lift."""
    base = history[0].get(key)
    final = history[-1].get(key)
    if base is None or final is None or final <= base:
        return None
    target = base + 0.5 * (final - base)
    for row in history:
        if row.get(key) is not None and row[key] >= target:
            return row["step"]
    return None


def _peak(history: list[dict], key: str) -> tuple[float, int]:
    """Max acc over training and the step it occurred (catches a transient
    lift that later collapses — e.g. the answer-leak teacher)."""
    best, best_s = -1.0, -1
    for row in history:
        v = row.get(key)
        if v is not None and v > best:
            best, best_s = v, row["step"]
    return best, best_s


def summarize(records: dict[str, dict]) -> str:
    out = ["", "=== Phase 2: OPD master table ==="]
    cfg = next(iter(records.values()))["config"]
    out.append(
        f"model={cfg['model']} loss={cfg['loss']} lr={cfg['lr']} "
        f"n_steps={cfg['n_steps']} bs={cfg['batch_size']} "
        f"max_new={cfg['max_new_tokens']} n_train={cfg['n_train']} "
        f"eval_sizes={cfg.get('eval_sizes')}"
    )
    hdr = (f"{'teacher':<11} {'tier':<18} {'eval':<18} "
           f"{'base':>6} {'final':>6} {'Δ':>7} {'peak':>6} {'peak@':>6} "
           f"{'½-lift@':>8}")
    out.append("")
    out.append(hdr)
    out.append("-" * len(hdr))
    for t in TEACHERS:
        rec = records.get(t)
        if rec is None:
            out.append(f"{t:<11} {'(missing)':<18}")
            continue
        h = rec["history"]
        for key, label in EVAL_SETS:
            base, final = h[0].get(key), h[-1].get(key)
            if base is None or final is None:
                continue
            hl = _half_lift_step(h, key)
            pk, pk_s = _peak(h, key)
            out.append(
                f"{t:<11} {TEACHER_TIER[t]:<18} {label:<18} "
                f"{base:>6.3f} {final:>6.3f} {final - base:>+7.3f} "
                f"{pk:>6.3f} {pk_s:>6} "
                f"{(str(hl) if hl is not None else '—'):>8}"
            )
        out.append("-" * len(hdr))
    return "\n".join(out)


def markdown_table(records: dict[str, dict]) -> str:
    rows = ["| teacher | tier | eval | base | final | Δ | peak | peak@ | ½-lift@ |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in TEACHERS:
        rec = records.get(t)
        if rec is None:
            continue
        h = rec["history"]
        for key, label in EVAL_SETS:
            base, final = h[0].get(key), h[-1].get(key)
            if base is None or final is None:
                continue
            hl = _half_lift_step(h, key)
            pk, pk_s = _peak(h, key)
            rows.append(
                f"| {t} | {TEACHER_TIER[t]} | {label} | {base:.3f} | "
                f"{final:.3f} | {final - base:+.3f} | {pk:.3f} | {pk_s} | "
                f"{hl if hl is not None else '—'} |"
            )
    return "\n".join(rows)


def plot_curves(records: dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, label) in zip(axes, EVAL_SETS):
        for t in TEACHERS:
            rec = records.get(t)
            if rec is None:
                continue
            h = rec["history"]
            xs = [r["step"] for r in h if r.get(key) is not None]
            ys = [r[key] for r in h if r.get(key) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
                    color=TEACHER_COLOR.get(t, "gray"),
                    label=f"{t} ({TEACHER_TIER[t]})")
        ax.set_title(label)
        ax.set_xlabel("OPD step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass@1 accuracy")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Phase 2: on-policy distillation learning curves by teacher context "
        "— leak vs transfer (ID vs OOD)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--md", action="store_true", help="emit markdown table")
    args = ap.parse_args()

    if args.pull:
        print(f"Pulled {len(pull_all())} files to {LOCAL_DIR}")

    records = load_local()
    if not records:
        print(f"No local results in {LOCAL_DIR}. Run --pull.", file=sys.stderr)
        return

    print(summarize(records))
    if args.md:
        print("\n" + markdown_table(records))
    if args.plot:
        plot_curves(records, FIG_PATH)


if __name__ == "__main__":
    main()
