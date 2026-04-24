import argparse
import json
from pathlib import Path

import pandas as pd

from model_features import REQUIRED_COLUMNS, coerce_explicit_to_bool


def build_payload_from_csv(csv_path, row_limit=None):
    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required fields: {missing_columns}")

    request_frame = frame[REQUIRED_COLUMNS].copy()
    if row_limit is not None:
        request_frame = request_frame.head(row_limit)

    if request_frame.empty:
        raise ValueError("The selected input rows are empty.")

    request_frame["explicit"] = request_frame["explicit"].apply(coerce_explicit_to_bool)
    records = request_frame.where(pd.notna(request_frame), None).to_dict(orient="records")
    return {"instances": records}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a training-format CSV into the JSON payload expected by the API."
    )
    parser.add_argument("input_csv", help="Path or URL to a CSV with the model training schema.")
    parser.add_argument("output_json", help="Path to write the API request payload.")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional number of top rows to export.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for the output JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_payload_from_csv(args.input_csv, row_limit=args.rows)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(payload, indent=args.indent)}\n", encoding="utf-8")

    print(f"Wrote {len(payload['instances'])} records to {output_path}")


if __name__ == "__main__":
    main()
