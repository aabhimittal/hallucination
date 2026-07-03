#!/usr/bin/env python3
"""Deploy the VERITAS demo to a Hugging Face Space.

    python scripts/deploy_space.py --repo <user>/veritas-hallucination-reduction
    python scripts/deploy_space.py --repo <user>/veritas-demo --token hf_xxx

Token resolution: --token flag, then HF_TOKEN env var, then the cached
`huggingface-cli login` credential. Needs a token with write access.

Uploads exactly what the Space needs: README.md (Spaces YAML frontmatter),
app.py, requirements.txt, src/veritas, benchmarks (corpus + dataset +
results), and the skill.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = [
    "README.md",
    "app.py",
    "requirements.txt",
    "LICENSE",
    "src/veritas",
    "benchmarks/corpus",
    "benchmarks/dataset.json",
    "benchmarks/results.md",
    "benchmarks/results.json",
    "benchmarks/run_benchmark.py",
    "skills/hallucination-reduction/SKILL.md",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Space id, e.g. user/veritas-demo")
    parser.add_argument("--token", default=None, help="HF write token (or set HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub  (or: pip install 'veritas-rag[hf]')")

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    identity = api.whoami()
    print(f"Authenticated as: {identity['name']}")

    api.create_repo(
        repo_id=args.repo,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    print(f"Space ready: https://huggingface.co/spaces/{args.repo}")

    # stage only the files the Space needs, preserving relative paths
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for rel in INCLUDE:
            src = ROOT / rel
            if not src.exists():
                print(f"  skipping missing {rel}")
                continue
            dest = staging / rel
            if src.is_dir():
                import shutil

                shutil.copytree(
                    src, dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
        api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy VERITAS demo",
        )
    print(f"Deployed. The Space is building at: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
