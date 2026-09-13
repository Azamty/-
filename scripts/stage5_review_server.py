"""Serve a completed stage5 job root so the browser can review its pages."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from backend.app import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(create_app(jobs_root=args.jobs_root), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
