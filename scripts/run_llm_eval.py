"""Provider-agnostic LLM evaluation harness for the BFO definitions experiment.

Reproduces the five experimental conditions (``1B_bare_masked_ids``,
``1A_bare_labels_only``, ``2A_sloppy_defs_only``, ``2B_strict_defs_only``,
``3_reasoner_plus_strict``) plus the optional ``04_clash`` condition across
Reason-First RAG and Conservative Abstain prompting modes.

Uses Python standard library ``urllib.request`` for OpenAI (and
OpenAI-compatible local servers such as Ollama/vLLM/LM Studio), Gemini, and
Anthropic APIs, with a lazy ``mlx_lm`` import only when ``--provider mlx`` is
selected. Condition ``3_reasoner_plus_strict`` dynamically materializes OWL-RL
inferences via ``verify_symbolic.evaluate_all()``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.verify_symbolic import evaluate_all
except ImportError:
    from verify_symbolic import evaluate_all

REPO_ROOT = Path(__file__).resolve().parent.parent

# Exact class-ID masking map used in Condition 1B (opaque identifier ablation).
MASK_MAP: dict[str, str] = {
    "SupplyChainRole": "CLASS_01",
    "ManufacturerRole": "CLASS_02",
    "BrokerRole": "CLASS_03",
    "CarrierRole": "CLASS_04",
    "InspectedMaterial": "CLASS_05",
    "ProductionBatch": "CLASS_06",
    "ForensicSpecimen": "CLASS_07",
    "AssayRunTray": "CLASS_08",
    "InspectionRecord": "CLASS_09",
    "ShippingManifest": "CLASS_10",
    "SpectroscopyDatum": "CLASS_11",
    "QuarantineDirective": "CLASS_12",
}

MASK_LABEL_MAP: dict[str, str] = {
    "Supply Chain Role": "Class 01",
    "Manufacturer Role": "Class 02",
    "Broker Role": "Class 03",
    "Carrier Role": "Class 04",
    "Inspected Material": "Class 05",
    "Production Batch": "Class 06",
    "Forensic Specimen": "Class 07",
    "Assay Run Tray": "Class 08",
    "Inspection Record": "Class 09",
    "Shipping Manifest": "Class 10",
    "Spectroscopy Datum": "Class 11",
    "Quarantine Directive": "Class 12",
}

SYSTEM_PROMPTS: dict[str, str] = {
    "rag": (
        "You are an enterprise knowledge-graph assistant answering competency "
        "questions from an RDF/OWL Turtle file. Reason briefly over the provided "
        "ontology, definitions, and facts, then return ONLY valid JSON with keys "
        'in this exact order: {"reason": "1-2 short sentences explaining your '
        'classification", "matches": ["EntityLocalId", ...]}. Use bare entity '
        'local IDs without colons (e.g., "Org_Vanguard_Works", "Lot_FX204", '
        '"Record_44B").'
    ),
    "abstain": (
        "You are an ontology QA evaluator inspecting an RDF/OWL Turtle file. "
        "Evaluate strictly against the given ontology definitions, axioms, and "
        'facts. Return ONLY valid JSON with two keys: {"matches": '
        '["EntityLocalId", ...], "reason": "one short sentence"}. Use bare '
        'entity local IDs without colons (for example "Org_Vanguard_Works" or '
        '"Lot_FX204"). If the provided knowledge does not deterministically '
        'identify any qualifying entity, return {"matches": [], "reason": "..."}.'
    ),
}

CQ_TARGETS: dict[str, str] = {
    "CQ1": "ManufacturerRole",
    "CQ2": "ProductionBatch",
    "CQ3": "SpectroscopyDatum",
}

CQ_EXPECTED: dict[str, list[str]] = {
    "CQ1": ["Org_Vanguard_Works"],
    "CQ2": ["Lot_FX204"],
    "CQ3": ["Record_44B"],
}

CQ_TEMPLATES: dict[str, dict[str, str]] = {
    "rag": {
        "CQ1": (
            "Which organization(s) among `:Org_Helios_Supply`, "
            "`:Org_Vanguard_Works`, and `:Org_Meridian_Freight` fit `:{target}` "
            "via their borne role? Use the provided ontology definitions, axioms, "
            "and graph facts to classify the entities."
        ),
        "CQ2": (
            "Which inspected material(s) among `:Lot_FX204`, `:Unit_FX204_A`, "
            "and `:Batch_Lab09` fit `:{target}`? Use the provided ontology "
            "definitions, axioms, and graph facts to classify the entities."
        ),
        "CQ3": (
            "Which inspection record(s) among `:Record_88A` and `:Record_44B` "
            "fit `:{target}`? Use the provided ontology definitions, axioms, and "
            "graph facts to classify the entities."
        ),
    },
    "abstain": {
        "CQ1": (
            "List every `:Organization` (`:Org_Helios_Supply`, "
            "`:Org_Vanguard_Works`, `:Org_Meridian_Freight`) whose borne role "
            "qualifies as `:{target}` strictly based on the provided ontology and "
            "consignment facts."
        ),
        "CQ2": (
            "List every inspected material (`:Lot_FX204`, `:Unit_FX204_A`, "
            "`:Batch_Lab09`) that qualifies as `:{target}` strictly based on the "
            "provided ontology and consignment facts."
        ),
        "CQ3": (
            "List every inspection record (`:Record_88A`, `:Record_44B`) that "
            "qualifies as `:{target}` strictly based on the provided ontology and "
            "consignment facts."
        ),
    },
}

STANDARD_CONDITIONS: tuple[str, ...] = (
    "1B_bare_masked_ids",
    "1A_bare_labels_only",
    "2A_sloppy_defs_only",
    "2B_strict_defs_only",
    "3_reasoner_plus_strict",
)

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
}


@dataclass(frozen=True)
class PromptSpec:
    """Single prompt payload ready for model evaluation."""

    mode: str
    condition: str
    cq_id: str
    system_prompt: str
    user_prompt: str
    question: str
    expected: list[str]


def mask_ontology_turtle(ttl_text: str) -> str:
    """Replace the 12 target domain class IRIs and labels with opaque CLASS_XX IDs."""
    masked = ttl_text
    for orig_id, masked_id in MASK_MAP.items():
        masked = re.sub(rf":{re.escape(orig_id)}\b", f":{masked_id}", masked)
    for orig_label, masked_label in MASK_LABEL_MAP.items():
        masked = masked.replace(f'"{orig_label}"', f'"{masked_label}"')
    return masked


def materialize_reasoner_turtle(repo_root: Path = REPO_ROOT) -> str:
    """Build Condition 3 Turtle with dynamically materialized OWL-RL types."""
    ont_file = "03_full_bfo_axioms_and_strict_defs.ttl"
    base_ttl = (repo_root / "ontologies" / ont_file).read_text(encoding="utf-8")
    symbolic_report = evaluate_all(repo_root)
    inferred_types: dict[str, list[str]] = symbolic_report["variants"][ont_file][
        "inferred_types"
    ]

    lines = [
        "",
        "### =========================================================",
        "### Materialized OWL-RL Reasoner Inferences (Inferred rdf:type)",
        "### =========================================================",
        "",
    ]
    for ind_id, classes in inferred_types.items():
        if classes:
            cls_str = " , ".join(f":{cls}" for cls in classes)
            lines.append(f":{ind_id} a {cls_str} .")

    return base_ttl.rstrip() + "\n" + "\n".join(lines) + "\n"


def load_condition_contexts(
    repo_root: Path = REPO_ROOT,
    include_clash: bool = False,
) -> dict[str, str]:
    """Load and prepare Turtle contexts for all requested conditions."""
    ontologies_dir = repo_root / "ontologies"
    bare_ttl = (ontologies_dir / "01_bare_no_defs_no_axioms.ttl").read_text(encoding="utf-8")

    contexts: dict[str, str] = {
        "1B_bare_masked_ids": mask_ontology_turtle(bare_ttl),
        "1A_bare_labels_only": bare_ttl,
        "2A_sloppy_defs_only": (
            ontologies_dir / "02a_sloppy_defs_no_axioms.ttl"
        ).read_text(encoding="utf-8"),
        "2B_strict_defs_only": (
            ontologies_dir / "02b_strict_defs_no_axioms.ttl"
        ).read_text(encoding="utf-8"),
        "3_reasoner_plus_strict": materialize_reasoner_turtle(repo_root),
    }

    if include_clash:
        contexts["04_clash"] = (
            ontologies_dir / "04_clash_manifest_as_spectroscopy.ttl"
        ).read_text(encoding="utf-8")

    return contexts


def build_question(mode: str, condition: str, cq_id: str) -> str:
    """Format the competency question wording for a mode/condition/CQ triple."""
    target_cls = CQ_TARGETS[cq_id]
    if condition == "1B_bare_masked_ids":
        target_cls = MASK_MAP[target_cls]
    return CQ_TEMPLATES[mode][cq_id].format(target=target_cls)


def build_user_prompt(ttl_text: str, question: str) -> str:
    """Combine ontology Turtle context and competency question into a user message."""
    return (
        "ONTOLOGY & CONSIGNMENT FACTS (Turtle):\n"
        "```turtle\n"
        f"{ttl_text.strip()}\n"
        "```\n\n"
        f"COMPETENCY QUESTION:\n{question}"
    )


def build_prompt_plan(
    modes: list[str],
    include_clash: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[PromptSpec]:
    """Build the ordered list of PromptSpecs across modes, conditions, and CQs."""
    contexts = load_condition_contexts(repo_root=repo_root, include_clash=include_clash)
    plan: list[PromptSpec] = []

    for mode in modes:
        system_prompt = SYSTEM_PROMPTS[mode]
        for condition, ttl_text in contexts.items():
            cq_ids = ("CQ3",) if condition == "04_clash" else ("CQ1", "CQ2", "CQ3")
            for cq_id in cq_ids:
                question = build_question(mode, condition, cq_id)
                plan.append(
                    PromptSpec(
                        mode=mode,
                        condition=condition,
                        cq_id=cq_id,
                        system_prompt=system_prompt,
                        user_prompt=build_user_prompt(ttl_text, question),
                        question=question,
                        expected=list(CQ_EXPECTED[cq_id]),
                    )
                )
    return plan


def parse_model_json(raw_text: str) -> tuple[list[str], str]:
    """Extract normalized ``matches`` and ``reason`` from raw LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fence_match.group(1) if fence_match else cleaned

    parsed: dict[str, Any] | None = None
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            parsed = obj
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(obj, dict) and ("matches" in obj or "reason" in obj):
                    parsed = obj
            except json.JSONDecodeError:
                continue

    if parsed is None:
        return [], f"[parse_error] {cleaned[:200]}"

    raw_matches = parsed.get("matches", [])
    if isinstance(raw_matches, str):
        raw_matches = [raw_matches]
    matches: list[str] = []
    if isinstance(raw_matches, list):
        for item in raw_matches:
            norm = str(item).strip()
            if "#" in norm:
                norm = norm.rsplit("#", 1)[-1]
            norm = norm.lstrip(":")
            if norm and norm not in matches:
                matches.append(norm)

    reason = str(parsed.get("reason", "")).strip()
    return matches, reason


def _http_post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Perform a JSON POST request using Python standard library ``urllib``."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {err_body}") from exc


class ProviderClient:
    """Thin provider-agnostic wrapper around OpenAI, Gemini, Anthropic, and MLX."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout
        self._mlx_loaded: tuple[Any, Any] | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch a single chat completion call to the configured provider."""
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        if self.provider == "mlx":
            return self._call_mlx(system_prompt, user_prompt)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        base = (self.base_url or DEFAULT_BASE_URLS["openai"]).rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key and not self.base_url:
            raise RuntimeError("OPENAI_API_KEY is required when using --provider openai.")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = _http_post_json(f"{base}/chat/completions", headers, payload, self.timeout)
        return str(data["choices"][0]["message"]["content"])

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is required when using --provider gemini."
            )
        base = (self.base_url or DEFAULT_BASE_URLS["gemini"]).rstrip("/")
        url = f"{base}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }
        data = _http_post_json(url, headers, payload, self.timeout)
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(str(part.get("text", "")) for part in parts)

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when using --provider anthropic."
            )
        base = (self.base_url or DEFAULT_BASE_URLS["anthropic"]).rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = _http_post_json(f"{base}/messages", headers, payload, self.timeout)
        blocks = data.get("content", [])
        return "".join(
            str(block.get("text", "")) for block in blocks if block.get("type") == "text"
        )

    def _call_mlx(self, system_prompt: str, user_prompt: str) -> str:
        from mlx_lm import generate, load

        if self._mlx_loaded is None:
            self._mlx_loaded = load(self.model)
        model_obj, tokenizer = self._mlx_loaded

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        kwargs: dict[str, Any] = {"max_tokens": 512, "verbose": False}
        try:
            from mlx_lm.sample_utils import make_sampler

            kwargs["sampler"] = make_sampler(temp=self.temperature)
        except ImportError:
            kwargs["temp"] = self.temperature

        return str(generate(model_obj, tokenizer, prompt=prompt_text, **kwargs))


def run_evaluation(
    modes: list[str],
    repeats: int,
    provider: str,
    model: str,
    temperature: float = 0.0,
    base_url: str | None = None,
    include_clash: bool = False,
    timeout: int = 120,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the full benchmark plan against a live model provider."""
    plan = build_prompt_plan(modes=modes, include_clash=include_clash, repo_root=repo_root)
    client = ProviderClient(
        provider=provider,
        model=model,
        temperature=temperature,
        base_url=base_url,
        timeout=timeout,
    )

    results: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "repeats": repeats,
        "modes": {},
    }

    for spec in plan:
        mode_bucket = results["modes"].setdefault(spec.mode, {})
        cond_bucket = mode_bucket.setdefault(
            spec.condition,
            {"cqs": {}, "total_exact": 0, "total_runs": 0, "accuracy_pct": 0.0},
        )
        runs: list[dict[str, Any]] = []
        exact_count = 0

        for run_idx in range(1, repeats + 1):
            t0 = time.perf_counter()
            raw = client.complete(spec.system_prompt, spec.user_prompt)
            latency_s = round(time.perf_counter() - t0, 2)
            matches, reason = parse_model_json(raw)
            exact = set(matches) == set(spec.expected)
            if exact:
                exact_count += 1
            false_positives = sorted(set(matches) - set(spec.expected))
            runs.append(
                {
                    "run": run_idx,
                    "temp": temperature,
                    "latency_s": latency_s,
                    "matches": matches,
                    "expected": spec.expected,
                    "exact_match": exact,
                    "false_positives": false_positives,
                    "reason": reason,
                    "raw": raw,
                }
            )

        cond_bucket["cqs"][spec.cq_id] = {
            "question": spec.question,
            "expected": spec.expected,
            "exact_count": exact_count,
            "runs_count": repeats,
            "accuracy_pct": round(100.0 * exact_count / repeats, 1) if repeats else 0.0,
            "runs": runs,
        }
        cond_bucket["total_exact"] += exact_count
        cond_bucket["total_runs"] += repeats
        cond_bucket["accuracy_pct"] = (
            round(100.0 * cond_bucket["total_exact"] / cond_bucket["total_runs"], 1)
            if cond_bucket["total_runs"]
            else 0.0
        )

    return results


def _print_dry_run_summary(
    plan: list[PromptSpec],
    repeats: int,
    provider: str,
    model: str,
) -> None:
    """Print prompt dimensions and verification stats without calling any API."""
    total_calls = len(plan) * repeats
    print(f"Dry-Run Evaluation Plan ({provider} / {model})")
    print("=" * 84)
    print(f"{'Mode':8s} {'Condition':26s} {'CQ':4s} {'SysChars':9s} {'UserChars':10s} {'Target'}")
    print("-" * 84)
    for spec in plan:
        target = CQ_TARGETS[spec.cq_id]
        if spec.condition == "1B_bare_masked_ids":
            target = MASK_MAP[target]
        print(
            f"{spec.mode:8s} {spec.condition:26s} {spec.cq_id:4s} "
            f"{len(spec.system_prompt):9d} {len(spec.user_prompt):10d} :{target}"
        )
    print("=" * 84)
    print(
        f"Prepared {len(plan)} unique prompts x {repeats} repeat(s) = "
        f"{total_calls} planned model calls (0 API calls executed)."
    )


def _print_live_summary(results: dict[str, Any]) -> None:
    """Print a concise accuracy table after a live evaluation run."""
    print(f"\nEvaluation Summary ({results['provider']} / {results['model']})")
    print("=" * 72)
    for mode, conds in results["modes"].items():
        print(f"Mode: {mode}")
        for cond, data in conds.items():
            cq_bits = " ".join(
                f"{cq}:{v['exact_count']}/{v['runs_count']}" for cq, v in data["cqs"].items()
            )
            print(
                f"  {cond:26s} {data['total_exact']:2d}/{data['total_runs']:2d} "
                f"({data['accuracy_pct']:5.1f}%)  {cq_bits}"
            )
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the 5-condition BFO definitions LLM evaluation benchmark.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini", "anthropic", "mlx"),
        default="openai",
        help="Model backend provider (default: openai).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (required unless --dry-run is set).",
    )
    parser.add_argument(
        "--mode",
        choices=("rag", "abstain", "both"),
        default="rag",
        help="Prompting regime to evaluate (default: rag).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated trials per competency question (default: 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional base URL (e.g. http://localhost:11434/v1 for Ollama).",
    )
    parser.add_argument(
        "--include-clash",
        action="store_true",
        help="Also include the 04_clash_manifest_as_spectroscopy CQ3 trial.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path to save full JSON traces.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build all prompts and print plan stats without making API calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1.")

    modes = ["rag", "abstain"] if args.mode == "both" else [args.mode]

    if args.dry_run:
        model_label = args.model or "dry-run"
        plan = build_prompt_plan(
            modes=modes,
            include_clash=args.include_clash,
            repo_root=REPO_ROOT,
        )
        _print_dry_run_summary(plan, args.repeats, args.provider, model_label)
        return 0

    if not args.model:
        parser.error("--model is required unless --dry-run is specified.")

    results = run_evaluation(
        modes=modes,
        repeats=args.repeats,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        include_clash=args.include_clash,
        timeout=args.timeout,
        repo_root=REPO_ROOT,
    )
    _print_live_summary(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved evaluation results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
