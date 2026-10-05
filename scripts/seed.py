"""Seed the index with a small sample corpus (Hybrid RAG).

    python scripts/seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.pipeline import get_pipeline  # noqa: E402

DOCS = [
    {"source_id": "error_codes.md", "text": (
        "# Error Codes\n\nERR-4032 indicates the authentication token has expired. "
        "Re-issue the token and retry the request. This error is returned by the gateway "
        "before the request reaches any backend service.\n\n"
        "ERR-5001 indicates a downstream timeout. Check the dependency health dashboard.")},
    {"source_id": "cost_guide.md", "text": (
        "# Reducing Cloud Spend\n\nOrganisations lower their cloud expenditure by "
        "right-sizing instances, purchasing commitment discounts and deleting orphaned "
        "volumes. Moving cold data to archival storage tiers also trims the monthly bill.")},
    {"source_id": "scaling.md", "text": (
        "# Autoscaling\n\nElastic capacity follows demand so you never pay for idle "
        "servers. Horizontal autoscaling adds instances under load and removes them when "
        "traffic subsides.")},
]


def main() -> None:
    p = get_pipeline()
    for d in DOCS:
        r = p.ingest_text(d["text"], source_id=d["source_id"])
        print(f"  ingested {r.source_id:20s} -> {r.num_chunks} chunk(s)")
    print(f"\nDone. chunks={p.stats()['chunks']} fusion={p.stats()['fusion']}")


if __name__ == "__main__":
    main()
