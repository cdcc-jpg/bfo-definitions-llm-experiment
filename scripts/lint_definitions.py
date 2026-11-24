"""SRS-17 textual definition linter for precursor-triage ontologies.

Audits every domain class carrying an ``obo:IAO_0000115`` definition against
three operationalized rules from Seppala, Ruttenberg, & Smith (2017):
``Circular``, ``Examples``, and ``Hedge``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rdflib import Graph, OWL, RDF, RDFS, URIRef

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_NS = "http://leurn.co/ontologies/precursor-triage#"
IAO_DEFINITION = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")

DEFAULT_FILES: tuple[str, ...] = (
    "02a_sloppy_defs_no_axioms.ttl",
    "02b_strict_defs_no_axioms.ttl",
    "03_full_bfo_axioms_and_strict_defs.ttl",
)

# Note: "e.g." ends with a non-word character, so trailing \b is applied only
# to the alphabetic phrases to avoid missing "(e.g., ...)".
EXAMPLES_RE = re.compile(
    r"\b(?:e\.g\.|for example\b|such as\b|including\b)",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(
    r"\b(?:typically|usually|often|generally|commonly|mostly|sometimes)\b",
    re.IGNORECASE,
)


def _check_srs17_rules(label: str, definition: str) -> list[str]:
    """Return sorted SRS-17 violation tags for a single class definition."""
    flags: list[str] = []
    if label and re.search(rf"\b{re.escape(label)}\b", definition, re.IGNORECASE):
        flags.append("Circular")
    if EXAMPLES_RE.search(definition):
        flags.append("Examples")
    if HEDGE_RE.search(definition):
        flags.append("Hedge")
    return sorted(flags)


def audit_file(ttl_path: Path) -> dict[str, Any]:
    """Audit all ``obo:IAO_0000115`` definitions of domain classes in a Turtle file.

    Args:
        ttl_path: Path to an RDF/OWL Turtle ontology file.

    Returns:
        Dictionary containing ``file``, ``total_audited``, ``flagged_count``,
        and per-term ``terms`` entries sorted by local IRI.
    """
    ttl_path = Path(ttl_path)
    graph = Graph()
    graph.parse(ttl_path, format="turtle")

    domain_classes = sorted(
        {
            subj
            for subj in graph.subjects(IAO_DEFINITION, None)
            if isinstance(subj, URIRef)
            and str(subj).startswith(DOMAIN_NS)
            and (subj, RDF.type, OWL.Class) in graph
        },
        key=str,
    )

    terms: list[dict[str, Any]] = []
    for cls_uri in domain_classes:
        local_id = str(cls_uri).rsplit("#", 1)[-1]
        label_node = graph.value(cls_uri, RDFS.label)
        label = str(label_node).strip() if label_node is not None else local_id

        definitions = sorted(str(obj).strip() for obj in graph.objects(cls_uri, IAO_DEFINITION))
        def_text = definitions[0] if definitions else ""
        flags = _check_srs17_rules(label, def_text)

        terms.append(
            {
                "iri": local_id,
                "label": label,
                "definition": def_text,
                "flags": flags,
            }
        )

    flagged_count = sum(1 for term in terms if term["flags"])
    return {
        "file": ttl_path.name,
        "total_audited": len(terms),
        "flagged_count": flagged_count,
        "terms": terms,
    }


def _format_audit(report: dict[str, Any]) -> str:
    """Format a single file audit report for CLI output."""
    header = (
        f"{report['file']}: "
        f"{report['flagged_count']}/{report['total_audited']} definitions flagged"
    )
    lines = [header, "-" * len(header)]
    flagged = [term for term in report["terms"] if term["flags"]]
    if not flagged:
        lines.append("  PASS: 0 SRS-17 violations detected.")
    else:
        for term in flagged:
            flag_str = ", ".join(term["flags"])
            lines.append(f"  [{flag_str}] :{term['iri']} ({term['label']})")
            lines.append(f"    \"{term['definition']}\"")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Audit ontology IAO:0000115 textual definitions against SRS-17 rules.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional Turtle file paths to audit (defaults to 02a, 02b, and 03).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    targets = args.paths or [REPO_ROOT / "ontologies" / name for name in DEFAULT_FILES]
    reports = [audit_file(path) for path in targets]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print("\n\n".join(_format_audit(report) for report in reports))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
