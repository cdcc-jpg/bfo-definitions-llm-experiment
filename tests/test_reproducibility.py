"""End-to-end reproducibility test suite for the BFO definitions & LLM experiment.

Verifies:
1. All five OWL 2 / BFO 2020 Turtle variants parse cleanly with expected counts.
2. Pure-Python OWL 2 RL symbolic reasoning reproduces 0/3 vs 3/3 CQ results,
   the 3-hop cascading material types, and the 04_clash disjointness error.
3. The SRS-17 Aristotelian definition linter flags exactly 5/12 sloppy
   definitions and 0/12 strict definitions.
4. Saved raw JSON trials independently match every published accuracy metric
   and contain zero leaked local machine paths.
5. The multi-provider LLM evaluation harness passes ``--dry-run`` cleanly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from rdflib import Graph, OWL, RDF, URIRef

from scripts.lint_definitions import audit_file
from scripts.summarize_results import summarize_all
from scripts.verify_symbolic import evaluate_all

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGIES_DIR = REPO_ROOT / "ontologies"
RESULTS_DIR = REPO_ROOT / "results"
DOMAIN_NS = "http://leurn.co/ontologies/precursor-triage#"
IAO_DEFINITION = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")

EXPECTED_TTL_FILES = (
    "01_bare_no_defs_no_axioms.ttl",
    "02a_sloppy_defs_no_axioms.ttl",
    "02b_strict_defs_no_axioms.ttl",
    "03_full_bfo_axioms_and_strict_defs.ttl",
    "04_clash_manifest_as_spectroscopy.ttl",
)

CORE_DOMAIN_CLASSES = frozenset(
    {
        "SupplyChainRole",
        "ManufacturerRole",
        "BrokerRole",
        "CarrierRole",
        "InspectedMaterial",
        "ProductionBatch",
        "ForensicSpecimen",
        "AssayRunTray",
        "InspectionRecord",
        "ShippingManifest",
        "SpectroscopyDatum",
        "QuarantineDirective",
    }
)

CONDITION_ORDER = (
    "1B_bare_masked_ids",
    "1A_bare_labels_only",
    "2A_sloppy_defs_only",
    "2B_strict_defs_only",
    "3_reasoner_plus_strict",
)


def _raw_condition_pct(cond_block: dict[str, Any]) -> float:
    """Recompute condition accuracy directly from raw ``runs`` arrays."""
    exact = 0
    total = 0
    for cq_data in cond_block["cqs"].values():
        for run in cq_data["runs"]:
            total += 1
            if set(run["matches"]) == set(run["expected"]):
                exact += 1
    assert total > 0
    return round(100.0 * exact / total, 1)


@pytest.mark.parametrize("filename", EXPECTED_TTL_FILES)
def test_ontologies_parse_and_have_expected_counts(filename: str) -> None:
    """Every Turtle variant parses cleanly with 12 core classes and 15 individuals."""
    ttl_path = ONTOLOGIES_DIR / filename
    assert ttl_path.is_file(), f"Missing ontology file: {filename}"

    graph = Graph()
    graph.parse(ttl_path, format="turtle")

    domain_classes = {
        str(subj).rsplit("#", 1)[-1]
        for subj in graph.subjects(RDF.type, OWL.Class)
        if isinstance(subj, URIRef) and str(subj).startswith(DOMAIN_NS)
    }
    assert CORE_DOMAIN_CLASSES.issubset(domain_classes)

    defined_domain_classes = {
        str(subj).rsplit("#", 1)[-1]
        for subj in graph.subjects(IAO_DEFINITION, None)
        if isinstance(subj, URIRef)
        and str(subj).startswith(DOMAIN_NS)
        and (subj, RDF.type, OWL.Class) in graph
    }
    if filename == "01_bare_no_defs_no_axioms.ttl":
        assert len(defined_domain_classes) == 0
    else:
        assert defined_domain_classes == CORE_DOMAIN_CLASSES
        assert len(defined_domain_classes) == 12

    individuals = {
        str(subj).rsplit("#", 1)[-1]
        for subj in graph.subjects(RDF.type, OWL.NamedIndividual)
        if isinstance(subj, URIRef) and str(subj).startswith(DOMAIN_NS)
    }
    assert len(individuals) == 15


def test_symbolic_owlrl_evaluation() -> None:
    """Verify 0/3 baseline, 3/3 full closure + 3-hop cascade, and 04_clash detection."""
    report = evaluate_all(REPO_ROOT)
    variants = report["variants"]
    assert tuple(sorted(variants.keys())) == EXPECTED_TTL_FILES

    # Conditions without classifying OWL axioms return 0/3 before and after reasoning
    for filename in (
        "01_bare_no_defs_no_axioms.ttl",
        "02a_sloppy_defs_no_axioms.ttl",
        "02b_strict_defs_no_axioms.ttl",
    ):
        v = variants[filename]
        assert v["consistent"] is True
        assert v["disjoint_clashes"] == []
        assert v["cq_before_inference"] == {"CQ1": [], "CQ2": [], "CQ3": []}
        assert v["cq_after_inference"] == {"CQ1": [], "CQ2": [], "CQ3": []}

    # Full BFO axioms + strict definitions: 0/3 before reasoning, 3/3 + 3-hop cascade after
    full = variants["03_full_bfo_axioms_and_strict_defs.ttl"]
    assert full["consistent"] is True
    assert full["disjoint_clashes"] == []
    assert full["cq_before_inference"] == {"CQ1": [], "CQ2": [], "CQ3": []}
    assert full["cq_after_inference"] == {
        "CQ1": ["Org_Vanguard_Works"],
        "CQ2": ["Lot_FX204"],
        "CQ3": ["Record_44B"],
    }
    inferred = full["inferred_types"]
    assert "ProductionBatch" in inferred["Lot_FX204"]
    assert "ForensicSpecimen" in inferred["Unit_FX204_A"]
    assert "AssayRunTray" in inferred["Batch_Lab09"]
    assert "ManufacturerRole" in inferred["Role_V1"]
    assert "BrokerRole" in inferred["Role_H1"]
    assert "CarrierRole" in inferred["Role_M1"]
    assert "ShippingManifest" in inferred["Record_88A"]
    assert "SpectroscopyDatum" in inferred["Record_44B"]

    # Disjointness clash on Record_88A between SpectroscopyDatum and ShippingManifest
    clash = variants["04_clash_manifest_as_spectroscopy.ttl"]
    assert clash["consistent"] is False
    assert any(
        entry["individual"] == "Record_88A"
        and {entry["class_a"], entry["class_b"]} == {"SpectroscopyDatum", "ShippingManifest"}
        for entry in clash["disjoint_clashes"]
    )


def test_srs17_definition_linter() -> None:
    """Verify SRS-17 flags exactly 5/12 sloppy definitions and 0/12 strict definitions."""
    sloppy = audit_file(ONTOLOGIES_DIR / "02a_sloppy_defs_no_axioms.ttl")
    assert sloppy["total_audited"] == 12
    assert sloppy["flagged_count"] == 5
    flagged_iris = {term["iri"] for term in sloppy["terms"] if term["flags"]}
    assert flagged_iris == {
        "ProductionBatch",
        "ManufacturerRole",
        "ShippingManifest",
        "AssayRunTray",
        "CarrierRole",
    }

    strict = audit_file(ONTOLOGIES_DIR / "02b_strict_defs_no_axioms.ttl")
    assert strict["total_audited"] == 12
    assert strict["flagged_count"] == 0

    full = audit_file(ONTOLOGIES_DIR / "03_full_bfo_axioms_and_strict_defs.ttl")
    assert full["total_audited"] == 12
    assert full["flagged_count"] == 0


def test_saved_results_integrity_and_published_metrics() -> None:
    """Verify all published percentages directly from raw runs and check path hygiene."""
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    expected_files = {
        "frontier_gemini_trials.json",
        "local_mlx_abstain_mode.json",
        "local_mlx_rag_mode.json",
    }
    assert expected_files.issubset({p.name for p in result_files})

    # Hygiene check: zero local /Users/ paths anywhere in results/*.json
    for json_path in result_files:
        raw_text = json_path.read_text(encoding="utf-8")
        assert "/Users/" not in raw_text, f"Local machine path leaked in {json_path.name}"

    summary = summarize_all(RESULTS_DIR)

    # 1. Qwen3.5-35B RAG mode
    rag_raw = json.loads((RESULTS_DIR / "local_mlx_rag_mode.json").read_text(encoding="utf-8"))
    qwen35_rag_expected = [0.0, 53.3, 93.3, 100.0, 100.0]
    assert [
        _raw_condition_pct(rag_raw["conditions"][c]) for c in CONDITION_ORDER
    ] == qwen35_rag_expected
    assert [
        summary["mlx_rag"]["Qwen3.5-35B-A3B-4bit"][c]["accuracy_pct"] for c in CONDITION_ORDER
    ] == qwen35_rag_expected

    # 2. Qwen3.5-35B & Qwen3-0.6B Abstain mode
    abstain_raw = json.loads(
        (RESULTS_DIR / "local_mlx_abstain_mode.json").read_text(encoding="utf-8")
    )
    qwen35_abstain_expected = [0.0, 0.0, 0.0, 60.0, 66.7]
    qwen06_abstain_expected = [0.0, 0.0, 0.0, 0.0, 0.0]

    assert [
        _raw_condition_pct(abstain_raw["models"]["Qwen3.5-35B-A3B-4bit"]["conditions"][c])
        for c in CONDITION_ORDER
    ] == qwen35_abstain_expected
    assert [
        summary["mlx_abstain"]["Qwen3.5-35B-A3B-4bit"][c]["accuracy_pct"]
        for c in CONDITION_ORDER
    ] == qwen35_abstain_expected

    assert [
        _raw_condition_pct(abstain_raw["models"]["Qwen3-0.6B-4bit"]["conditions"][c])
        for c in CONDITION_ORDER
    ] == qwen06_abstain_expected
    assert [
        summary["mlx_abstain"]["Qwen3-0.6B-4bit"][c]["accuracy_pct"] for c in CONDITION_ORDER
    ] == qwen06_abstain_expected

    # 3. Frontier Gemini Reasoning Model (RAG, Abstain, and 04_clash)
    gemini_raw = json.loads(
        (RESULTS_DIR / "frontier_gemini_trials.json").read_text(encoding="utf-8")
    )
    gemini_rag_expected = [0.0, 100.0, 33.3, 100.0, 100.0]
    gemini_abstain_expected = [0.0, 0.0, 0.0, 100.0, 100.0]

    assert [
        _raw_condition_pct(gemini_raw["modes"]["rag"][c]) for c in CONDITION_ORDER
    ] == gemini_rag_expected
    assert [
        summary["frontier_gemini"]["modes"]["rag"][c]["accuracy_pct"] for c in CONDITION_ORDER
    ] == gemini_rag_expected

    assert [
        _raw_condition_pct(gemini_raw["modes"]["abstain"][c]) for c in CONDITION_ORDER
    ] == gemini_abstain_expected
    assert [
        summary["frontier_gemini"]["modes"]["abstain"][c]["accuracy_pct"]
        for c in CONDITION_ORDER
    ] == gemini_abstain_expected

    clash_runs = gemini_raw["clash"]
    clash_exact = sum(
        1 for run in clash_runs if set(run["matches"]) == set(run["expected"])
    )
    assert len(clash_runs) == 3
    assert clash_exact == 3
    assert summary["frontier_gemini"]["clash"]["exact"] == 3
    assert summary["frontier_gemini"]["clash"]["total"] == 3
    assert summary["frontier_gemini"]["clash"]["accuracy_pct"] == 100.0


def test_run_llm_eval_dry_run_exits_cleanly() -> None:
    """Verify scripts/run_llm_eval.py --dry-run exits with code 0 without API keys."""
    script_path = REPO_ROOT / "scripts" / "run_llm_eval.py"
    assert script_path.is_file(), "Missing scripts/run_llm_eval.py"

    proc = subprocess.run(
        [sys.executable, str(script_path), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Dry-run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
