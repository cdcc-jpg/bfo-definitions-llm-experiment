"""Pure-Python OWL-RL symbolic verification for the precursor-triage ontologies.

Evaluates every Turtle variant in ``ontologies/`` against the three SPARQL
competency queries in ``queries/`` both before and after OWL 2 RL deductive
closure (via ``owlrl``), checks for ``owl:disjointWith`` inconsistencies, and
records inferred domain classes for the key ABox individuals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import owlrl
from rdflib import Graph, RDF, URIRef

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_NS = "http://leurn.co/ontologies/precursor-triage#"

CQ_QUERY_FILES: dict[str, str] = {
    "CQ1": "cq1_manufacturer_role.sparql",
    "CQ2": "cq2_production_batch.sparql",
    "CQ3": "cq3_spectroscopy_datum.sparql",
}

KEY_INDIVIDUALS: tuple[str, ...] = (
    "Role_H1",
    "Role_V1",
    "Role_M1",
    "Lot_FX204",
    "Unit_FX204_A",
    "Batch_Lab09",
    "Record_88A",
    "Record_44B",
)

DISJOINT_CLASH_QUERY = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
SELECT DISTINCT ?ind ?c1 ?c2 WHERE {
  ?ind a ?c1 , ?c2 .
  ?c1 owl:disjointWith ?c2 .
}
"""


def _local_id(term: Any) -> str:
    """Extract the local identifier from an IRI (splitting on ``#``, fallback ``/``)."""
    iri = str(term)
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rsplit("/", 1)[-1]


def _run_cqs(graph: Graph, queries: dict[str, str]) -> dict[str, list[str]]:
    """Execute all competency SPARQL queries and return sorted local IDs."""
    results: dict[str, list[str]] = {}
    for cq_id, sparql in queries.items():
        ids = {_local_id(row[0]) for row in graph.query(sparql) if row and row[0]}
        results[cq_id] = sorted(ids)
    return results


def _find_disjoint_clashes(graph: Graph) -> list[dict[str, str]]:
    """Detect and deduplicate ``owl:disjointWith`` clashes across named resources."""
    seen: set[tuple[str, str, str]] = set()
    for row in graph.query(DISJOINT_CLASH_QUERY):
        ind, c1, c2 = row[0], row[1], row[2]
        if not (isinstance(ind, URIRef) and isinstance(c1, URIRef) and isinstance(c2, URIRef)):
            continue
        ind_id = _local_id(ind)
        class_a, class_b = sorted((_local_id(c1), _local_id(c2)))
        seen.add((ind_id, class_a, class_b))
    return [
        {"individual": ind_id, "class_a": class_a, "class_b": class_b}
        for ind_id, class_a, class_b in sorted(seen)
    ]


def _extract_inferred_types(graph: Graph) -> dict[str, list[str]]:
    """Return sorted inferred domain-namespace types for each key individual."""
    inferred: dict[str, list[str]] = {}
    for ind_id in KEY_INDIVIDUALS:
        subject = URIRef(f"{DOMAIN_NS}{ind_id}")
        types = {
            _local_id(obj)
            for obj in graph.objects(subject, RDF.type)
            if isinstance(obj, URIRef) and str(obj).startswith(DOMAIN_NS)
        }
        inferred[ind_id] = sorted(types)
    return inferred


def evaluate_all(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Evaluate all ontology variants before and after OWL 2 RL reasoning.

    Args:
        repo_root: Repository root containing ``ontologies/`` and ``queries/``.

    Returns:
        Dictionary with top-level ``"variants"`` mapping each Turtle filename
        to consistency status, disjoint clashes, CQ answers before/after
        inference, and inferred domain types for key individuals.
    """
    repo_root = Path(repo_root)
    ontologies_dir = repo_root / "ontologies"
    queries_dir = repo_root / "queries"

    queries = {
        cq_id: (queries_dir / filename).read_text(encoding="utf-8")
        for cq_id, filename in CQ_QUERY_FILES.items()
    }

    variants: dict[str, Any] = {}
    for ttl_path in sorted(ontologies_dir.glob("*.ttl")):
        graph = Graph()
        graph.parse(ttl_path, format="turtle")

        cq_before = _run_cqs(graph, queries)
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(graph)
        cq_after = _run_cqs(graph, queries)
        clashes = _find_disjoint_clashes(graph)
        inferred_types = _extract_inferred_types(graph)

        variants[ttl_path.name] = {
            "consistent": len(clashes) == 0,
            "disjoint_clashes": clashes,
            "cq_before_inference": cq_before,
            "cq_after_inference": cq_after,
            "inferred_types": inferred_types,
        }

    return {"variants": variants}


def _format_summary(report: dict[str, Any]) -> str:
    """Render a crisp text table summarizing symbolic reasoning results."""
    lines = [
        "OWL-RL Symbolic Verification Summary",
        "=" * 92,
        f"{'Ontology Variant':44s} {'Consistent':10s} {'CQ1':18s} {'CQ2':10s} {'CQ3':18s}",
        "-" * 92,
    ]
    for filename, data in report["variants"].items():
        status = "YES" if data["consistent"] else "NO (CLASH)"
        after = data["cq_after_inference"]
        cq1 = ",".join(after["CQ1"]) or "[]"
        cq2 = ",".join(after["CQ2"]) or "[]"
        cq3 = ",".join(after["CQ3"]) or "[]"
        lines.append(f"{filename:44s} {status:10s} {cq1:18s} {cq2:10s} {cq3:18s}")
        for clash in data["disjoint_clashes"]:
            lines.append(
                f"  -> Clash: :{clash['individual']} a :{clash['class_a']} "
                f"(owl:disjointWith :{clash['class_b']})"
            )
    lines.append("=" * 92)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run pure-Python OWL 2 RL symbolic verification across all ontology variants.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write the full JSON report to results/symbolic_report.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    report = evaluate_all(REPO_ROOT)
    print(_format_summary(report))

    if args.write_report:
        report_path = REPO_ROOT / "results" / "symbolic_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote symbolic report to {report_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
