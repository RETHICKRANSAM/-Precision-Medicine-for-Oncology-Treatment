"""
Batch Scenario Uploader to Supabase for GenAI Stage.

Reads generated scenarios from genai/outputs/generated_scenarios.jsonl,
validates schema and integrity, and upserts them to public.generated_scenarios.
Does not alter original JSONL/CSV files or expose credentials.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genai.integration.supabase_client import get_supabase_client, test_connection


REQUIRED_FIELDS = [
    "scenario_id",
    "severity",
    "patient_scenario",
    "compound_interactions",
    "potential_risk_context",
    "seed_conditions",
    "generation_metadata",
    "validation"
]


def load_generated_scenarios(jsonl_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Reads and parses generated scenarios from JSONL file.
    Does not modify the original file.
    
    Returns:
        Tuple of (valid_records, parsing_errors)
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Error: Scenarios file not found at '{jsonl_path}'. Run evaluation first.")

    valid_records: List[Dict[str, Any]] = []
    parsing_errors: List[str] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                if not isinstance(record, dict):
                    parsing_errors.append(f"Line {line_num}: JSON content is not a dictionary object.")
                    continue
                valid_records.append(record)
            except json.JSONDecodeError as exc:
                parsing_errors.append(f"Line {line_num}: Malformed JSON syntax ({exc}).")

    return valid_records, parsing_errors


def prepare_upload_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and preserves required fields for the Supabase schema.
    """
    return {
        "scenario_id": str(record.get("scenario_id")),
        "severity": str(record.get("severity")),
        "patient_scenario": str(record.get("patient_scenario")),
        "compound_interactions": record.get("compound_interactions", []),
        "potential_risk_context": record.get("potential_risk_context", []),
        "seed_conditions": record.get("seed_conditions", {}),
        "generation_metadata": record.get("generation_metadata", {}),
        "validation": record.get("validation", {})
    }


def upload_scenarios_to_supabase(
    jsonl_path: str = "genai/outputs/generated_scenarios.jsonl",
    table_name: str = "generated_scenarios"
) -> Dict[str, Any]:
    """
    Uploads validated scenarios from JSONL into Supabase public.generated_scenarios.
    Uses upsert on scenario_id to prevent duplicates on repeated executions.
    
    Returns:
        Summary metrics dictionary.
    """
    print("=" * 70)
    print("SUPABASE SCENARIO UPLOAD PIPELINE")
    print("=" * 70)
    print(f"Target Table: public.{table_name}")
    print(f"Source File:  {jsonl_path}")

    # 1. Connection check
    is_connected, conn_msg = test_connection(table_name=table_name)
    if not is_connected:
        raise ConnectionError(f"Connection verification failed: {conn_msg}")
    print(f"[OK] {conn_msg}")

    # 2. Read scenarios
    records, parse_errors = load_generated_scenarios(jsonl_path)
    total_records = len(records)
    print(f"[OK] Read {total_records} records from JSONL file.")
    if parse_errors:
        print(f"[WARNING] Encountered {len(parse_errors)} parsing errors:")
        for err in parse_errors:
            print(f"  - {err}")

    # 3. Client initialization
    client = get_supabase_client()

    # Query existing scenario IDs in database to detect new vs updated (upserted)
    existing_ids = set()
    try:
        existing_res = client.table(table_name).select("scenario_id").execute()
        if existing_res.data:
            existing_ids = {row["scenario_id"] for row in existing_res.data if "scenario_id" in row}
    except Exception as exc:
        print(f"[NOTE] Could not pre-query existing IDs ({exc}); proceeding with upsert.")

    successful_uploads = 0
    failed_uploads = 0
    duplicate_upserted = 0
    invalid_records = 0
    errors_list = []

    # 4. Filter & upload records
    for record in records:
        scen_id = record.get("scenario_id")

        # Validation integrity check
        validation = record.get("validation", {})
        is_valid = validation.get("is_valid", True)
        if not is_valid:
            print(f"  [SKIP] Skipping invalid scenario '{scen_id}' (validation.is_valid is False).")
            invalid_records += 1
            continue

        # Check required schema fields
        missing_fields = [f for f in REQUIRED_FIELDS if f not in record or record[f] is None]
        if missing_fields:
            print(f"  [SKIP] Skipping scenario '{scen_id}': missing required fields {missing_fields}.")
            invalid_records += 1
            continue

        payload = prepare_upload_payload(record)
        is_duplicate = scen_id in existing_ids

        try:
            # Upsert using scenario_id as conflict resolution key
            response = client.table(table_name).upsert(payload, on_conflict="scenario_id").execute()
            if response.data or len(response.data) >= 0:
                successful_uploads += 1
                if is_duplicate:
                    duplicate_upserted += 1
                else:
                    existing_ids.add(scen_id)
            else:
                failed_uploads += 1
                errors_list.append((scen_id, "Empty response data on upsert."))
        except Exception as exc:
            failed_uploads += 1
            err_text = str(exc)
            errors_list.append((scen_id, err_text))
            print(f"  [ERROR] Failed to upload '{scen_id}': {err_text}")

    report = {
        "total_records": total_records,
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads,
        "duplicate_upserted_records": duplicate_upserted,
        "invalid_skipped_records": invalid_records,
        "parsing_errors": len(parse_errors)
    }

    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY REPORT")
    print("=" * 70)
    print(f"  - Total Records Evaluated:       {report['total_records']}")
    print(f"  - Successful Uploads:            {report['successful_uploads']}")
    print(f"  - Failed Uploads:                {report['failed_uploads']}")
    print(f"  - Duplicate / Upserted Records:  {report['duplicate_upserted_records']}")
    print(f"  - Invalid / Skipped Records:     {report['invalid_skipped_records']}")
    print(f"  - Syntax / Parsing Errors:       {report['parsing_errors']}")

    if failed_uploads > 0:
        print("\nErrors encountered during upload:")
        for sid, err in errors_list:
            print(f"  * {sid}: {err}")

    return report


if __name__ == "__main__":
    jsonl_target = "genai/outputs/generated_scenarios.jsonl"
    if len(sys.argv) > 1:
        jsonl_target = sys.argv[1]

    try:
        upload_scenarios_to_supabase(jsonl_path=jsonl_target)
    except Exception as e:
        print(f"\n[FATAL ERROR] Upload failed: {e}", file=sys.stderr)
        sys.exit(1)
