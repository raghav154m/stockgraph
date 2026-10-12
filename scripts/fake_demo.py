"""Build a small fake demo dataset so the API and dashboard run without the Kaggle download.

Same steps as the backend tests: fake Favorita -> prepare -> graph -> baseline pipeline ->
ST-GNN stand-in -> blend. Takes a minute or two. Output goes to data/fake/ (gitignored).

  python scripts/fake_demo.py
  STOCKGRAPH_DATA_DIR=data/fake/processed STOCKGRAPH_DB_URL=sqlite:///data/fake/stockgraph.db \
      uvicorn backend.main:app --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.prepare import prepare  # noqa: E402
from ml.graph.build import build_graph  # noqa: E402
from ml.models.ensemble import blend  # noqa: E402
from ml.pipeline import run  # noqa: E402
from tests.fake_favorita import write_fake_raw, write_fake_stgnn_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fake")
    args = parser.parse_args()

    processed = args.out / "processed"
    write_fake_raw(args.out / "raw", days=200)
    prepare(args.out / "raw", processed, days=180)
    build_graph(processed)
    run(processed, train_days=120, verbose=False)
    write_fake_stgnn_outputs(processed / "results")
    blend(processed, verbose=False)
    (args.out / "stockgraph.db").unlink(missing_ok=True)   # the API reloads the demo on next start
    print(f"Fake demo written to {processed}")


if __name__ == "__main__":
    main()
