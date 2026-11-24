"""Recompute and summarize benchmark accuracy tables from raw LLM run traces.

Reads ``local_mlx_rag_mode.json``, ``local_mlx_abstain_mode.json``, and
``frontier_gemini_trials.json`` from ``results/`` and independently recomputes
exact-match scores and false-positive counts directly from each run's
``matches`` and ``expected`` lists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

CONDITION_ORDER: tuple[str, ...] = (
    "1B_bare_masked_ids",
    "1A_bare_labels_only",
    "2A_sloppy_defs_only",
    "2B_strict_defs_only",
    "3_reasoner_plus_strict",
)

CQ_ORDER: tuple[str, ...] = ("CQ1", "CQ2", "CQ3")


def _pct(exact: int, total: int) -> float:
    """Compute a percentage rounded to one decimal place."""
    return round(100.0 * exact / total, 1) if total else 0.0


def _score_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute exact matches and false positives directly from raw runs."""
    exact = 0
    fp_counter: Counter[str] = Counter()
    for run in runs:
        predicted = set(run["matches"])
        expected = set(run["expected"])
        if predicted == expected:
            exact += 1
        for fp in sorted(predicted - expected):
            fp_counter[fp] += 1
    total = len(runs)
    return {
        "exact": exact,
        "total": total,
        "accuracy_pct": _pct(exact, total),
        "false_positives": dict(sorted(fp_counter.items())),
    }


def _ordered_conditions(conditions: dict[str, Any]) -> list[str]:
    """Return condition keys in canonical experimental progression order."""
    known = [cond for cond in CONDITION_ORDER if cond in conditions]
    extra = sorted(cond for cond in conditions if cond not in CONDITION_ORDER)
    return known + extra


def _summarize_condition_block(conditions: dict[str, Any]) -> dict[str, Any]:
    """Recompute per-condition and per-CQ metrics from a ``conditions`` mapping."""
    summary: dict[str, Any] = {}
    for cond in _ordered_conditions(conditions):
        cqs_raw = conditions[cond]["cqs"]
        cqs_summary: dict[str, Any] = {}
        total_exact = 0
        total_runs = 0
        cond_fps: Counter[str] = Counter()

        cq_keys = [cq for cq in CQ_ORDER if cq in cqs_raw] + sorted(
            cq for cq in cqs_raw if cq not in CQ_ORDER
        )
        for cq in cq_keys:
            scored = _score_runs(cqs_raw[cq]["runs"])
            cqs_summary[cq] = scored
            total_exact += scored["exact"]
            total_runs += scored["total"]
            for entity, count in scored["false_positives"].items():
                cond_fps[f"{cq}:{entity}"] += count

        summary[cond] = {
            "exact": total_exact,
            "total": total_runs,
            "accuracy_pct": _pct(total_exact, total_runs),
            "cqs": cqs_summary,
            "false_positives": dict(sorted(cond_fps.items())),
        }
    return summary


def summarize_all(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Any]:
    """Load all saved result JSONs and recompute exact-match and FP metrics.

    Args:
        results_dir: Directory containing the three benchmark result files.

    Returns:
        Structured dictionary with recomputed metrics for ``mlx_rag``,
        ``mlx_abstain``, and ``frontier_gemini`` (including ``clash``).
    """
    results_dir = Path(results_dir)

    rag_raw = json.loads((results_dir / "local_mlx_rag_mode.json").read_text(encoding="utf-8"))
    abstain_raw = json.loads(
        (results_dir / "local_mlx_abstain_mode.json").read_text(encoding="utf-8")
    )
    gemini_raw = json.loads(
        (results_dir / "frontier_gemini_trials.json").read_text(encoding="utf-8")
    )

    mlx_rag = {
        rag_raw["model"]: _summarize_condition_block(rag_raw["conditions"]),
    }

    mlx_abstain = {
        model_name: _summarize_condition_block(model_data["conditions"])
        for model_name, model_data in abstain_raw["models"].items()
    }

    gemini_modes = {
        mode: _summarize_condition_block(conds)
        for mode, conds in gemini_raw["modes"].items()
    }
    clash_runs = gemini_raw.get("clash", [])
    gemini_clash = _score_runs(clash_runs)

    return {
        "mlx_rag": mlx_rag,
        "mlx_abstain": mlx_abstain,
        "frontier_gemini": {
            "model": gemini_raw["model"],
            "modes": gemini_modes,
            "clash": gemini_clash,
        },
    }


def _fmt_cell(scored: dict[str, Any]) -> str:
    """Render an ``exact/total (pct%)`` cell."""
    return f"{scored['exact']}/{scored['total']} ({scored['accuracy_pct']:.1f}%)"


def _render_condition_table(title: str, model_label: str, conds: dict[str, Any]) -> list[str]:
    """Render a Markdown table for one model + prompt mode."""
    lines = [
        f"### {title} — {model_label}",
        "",
        "| Condition | CQ1 | CQ2 | CQ3 | Overall | False Positives |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for cond, data in conds.items():
        cq1 = f"{data['cqs']['CQ1']['exact']}/{data['cqs']['CQ1']['total']}"
        cq2 = f"{data['cqs']['CQ2']['exact']}/{data['cqs']['CQ2']['total']}"
        cq3 = f"{data['cqs']['CQ3']['exact']}/{data['cqs']['CQ3']['total']}"
        overall = _fmt_cell(data)
        fps = (
            ", ".join(f"{k} ({v})" for k, v in data["false_positives"].items())
            if data["false_positives"]
            else "—"
        )
        lines.append(f"| `{cond}` | {cq1} | {cq2} | {cq3} | {overall} | {fps} |")
    lines.append("")
    return lines


def format_markdown(summary: dict[str, Any]) -> str:
    """Format the full recomputed benchmark summary as Markdown."""
    lines: list[str] = ["## Recomputed Benchmark Results (from raw `runs` arrays)", ""]

    gemini = summary["frontier_gemini"]
    lines.extend(
        _render_condition_table("Reason-First RAG Mode", gemini["model"], gemini["modes"]["rag"])
    )
    for model_name, conds in summary["mlx_rag"].items():
        lines.extend(_render_condition_table("Reason-First RAG Mode", model_name, conds))

    lines.extend(
        _render_condition_table(
            "Conservative Abstain Mode", gemini["model"], gemini["modes"]["abstain"]
        )
    )
    for model_name, conds in summary["mlx_abstain"].items():
        lines.extend(_render_condition_table("Conservative Abstain Mode", model_name, conds))

    clash = gemini["clash"]
    lines.extend(
        [
            "### Disjointness Clash Trial (`04_clash_manifest_as_spectroscopy.ttl`)",
            "",
            "| Model | Mode | CQ3 Exact Match | False Positives |",
            "| :--- | :--- | :---: | :--- |",
            f"| {gemini['model']} | `clash_rag` | {_fmt_cell(clash)} | "
            f"{', '.join(f'{k} ({v})' for k, v in clash['false_positives'].items()) or '—'} |",
            "",
        ]
    )

    # Highlight key false-positive traces cleanly
    lines.extend(["### Key False-Positive Trace Counts", ""])
    fp_entries: list[str] = []

    def _collect_fps(label: str, conds: dict[str, Any]) -> None:
        for cond, data in conds.items():
            if data["false_positives"]:
                detail = ", ".join(f"`{k}` x{v}" for k, v in data["false_positives"].items())
                fp_entries.append(f"- **{label}** / `{cond}`: {detail}")

    _collect_fps("Frontier Gemini (rag)", gemini["modes"]["rag"])
    _collect_fps("Frontier Gemini (abstain)", gemini["modes"]["abstain"])
    for model_name, conds in summary["mlx_rag"].items():
        _collect_fps(f"{model_name} (rag)", conds)
    for model_name, conds in summary["mlx_abstain"].items():
        _collect_fps(f"{model_name} (abstain)", conds)

    lines.extend(fp_entries or ["- None."])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Recompute exact-match accuracy and false-positive tables from saved LLM runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to the results directory (default: results/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of Markdown tables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    summary = summarize_all(args.results_dir)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
