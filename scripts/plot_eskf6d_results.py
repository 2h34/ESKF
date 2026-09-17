"""Generate Phase 2.5 RPY submission data and figures from frozen results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_result import create_submission_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=PROJECT_ROOT / "results/eskf6d_result.csv",
    )
    parser.add_argument(
        "--pose",
        type=Path,
        default=PROJECT_ROOT / "data/processed/pose_processed.npz",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=PROJECT_ROOT / "data/processed/match_table.npz",
    )
    parser.add_argument(
        "--eskf-summary",
        type=Path,
        default=PROJECT_ROOT / "results/eskf6d_summary.json",
    )
    parser.add_argument(
        "--nis-summary",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/nis_analysis_summary.json",
    )
    parser.add_argument(
        "--dynamic-summary",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/dynamic_consistency_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    arguments = parser.parse_args()
    paths, summary = create_submission_results(
        arguments.result_csv,
        arguments.pose,
        arguments.alignment,
        arguments.eskf_summary,
        arguments.nis_summary,
        arguments.dynamic_summary,
        arguments.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for field, value in vars(paths).items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
