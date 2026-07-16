"""Index a legacy repository from the command line.

    .venv\\Scripts\\python.exe scripts\\index_repo.py D:\\path\\to\\repo
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexer import index_repository            # noqa: E402
from app.logging_setup import setup_logging         # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    setup_logging()
    summary = index_repository(sys.argv[1], user="cli")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
