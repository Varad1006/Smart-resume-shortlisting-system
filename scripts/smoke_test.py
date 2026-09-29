"""Upload the sample resumes to a running server and print the ranked shortlist.

uv run python scripts/smoke_test.py [--url http://localhost:8000] [--social] [--insights]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
LABELS = {
    "text_layer": "digital PDF",
    "docx": "Word",
    "plain_text": "text",
    "chandra": "scan",
    "tesseract": "scan",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--social", action="store_true", help="enable the public-profile bonus")
    parser.add_argument("--insights", action="store_true", help="request AI summaries (LLM)")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    job = (SAMPLES / "job_description.txt").read_text(encoding="utf-8")
    resumes = [p for p in sorted(SAMPLES.iterdir()) if p.name != "job_description.txt"]
    files = [("files", (p.name, p.read_bytes())) for p in resumes]

    with httpx.Client(base_url=args.url, timeout=120) as client:
        status = client.get("/api/v1/status").json()
        scans = any(e["available"] for e in status["ocr"]["engines"])
        ai = status.get("llm", {}).get("configured", False)
        print(
            f"Server {status['version']} | scans: {'ready' if scans else 'unavailable'} | "
            f"AI summaries: {'ready' if ai else 'off'}"
        )

        response = client.post(
            "/api/v1/shortlists",
            data={
                "job_description": job,
                "include_social": str(args.social).lower(),
                "include_insights": str(args.insights).lower(),
            },
            files=files,
        )
        response.raise_for_status()
        run_id = response.json()["id"]
        print(f"Uploaded {len(files)} resumes -> run {run_id}")

        deadline = time.monotonic() + args.timeout
        while True:
            run = client.get(f"/api/v1/shortlists/{run_id}").json()
            if run["status"] in ("completed", "failed"):
                break
            if time.monotonic() > deadline:
                sys.exit("Timed out waiting for the run to finish.")
            progress = run["progress"]
            print(f"  {progress['stage']:<11} {progress['done']}/{progress['total']}")
            time.sleep(2)

    if run["status"] == "failed":
        sys.exit(f"Run failed: {run['error']}")

    result = run["result"]
    print(f"\nRequirements: {len(result['requirements'])} | timings: {result['timings']}")
    header = (
        f"{'#':>3}  {'file':<34}{'lang':>5}  {'format':<24}"
        f"{'sem':>6}{'cov':>6}{'bonus':>6}{'final':>7}"
    )
    print(header)
    print("-" * len(header))
    for c in result["candidates"]:
        rank = c["final_rank"] or "-"
        methods = "+".join(dict.fromkeys(LABELS.get(m, m) for m in c["extraction_methods"]))
        print(
            f"{rank:>3}  {c['filename']:<34}{c['language'] or '?':>5}  {methods:<24}"
            f"{c['semantic_score']:6.1f}{c['coverage_score'] or 0:6.1f}{c['social_bonus']:6.1f}"
            f"{c['final_score'] or 0:7.1f}"
        )
    for c in result["candidates"]:
        if c.get("insight"):
            print(f"\nAI summary for {c['filename']}: {c['insight']['summary']}")
        elif c.get("insight_error"):
            print(f"\nAI summary for {c['filename']} failed: {c['insight_error']}")
    for skipped in result["skipped"]:
        print(f"skipped: {skipped['filename']}: {skipped['reason']}")
    print(f"\nOpen {args.url}/shortlists/{run_id} for the evidence view.")


if __name__ == "__main__":
    main()
