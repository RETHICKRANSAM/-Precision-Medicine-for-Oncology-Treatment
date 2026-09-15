"""
Verification Script for Supabase Uploaded Scenarios.

Queries public.generated_scenarios and displays row count and scenario summaries.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genai.integration.supabase_client import get_supabase_client


def verify_uploaded_scenarios(table_name: str = "generated_scenarios") -> None:
    """Queries and displays summary of records currently stored in Supabase."""
    client = get_supabase_client()
    
    # Get total count
    count_res = client.table(table_name).select("count", count="exact").execute()
    total_count = count_res.count if count_res.count is not None else 0
    
    # Fetch all records
    records_res = client.table(table_name).select("scenario_id, severity, patient_scenario, created_at").order("scenario_id").execute()
    records = records_res.data or []
    
    print("=" * 80)
    print(f"SUPABASE DATABASE VERIFICATION: public.{table_name}")
    print("=" * 80)
    print(f"Total Confirmed Records in Supabase: {total_count}\n")
    print(f"{'#':<3} {'Scenario ID':<36} {'Severity':<10} {'Scenario Preview':<30}")
    print("-" * 80)
    
    for idx, row in enumerate(records, start=1):
        sid = row.get("scenario_id", "")
        sev = row.get("severity", "")
        scen = row.get("patient_scenario", "").replace("\n", " ")[:28] + ".."
        print(f"{idx:<3} {sid:<36} {sev:<10} {scen:<30}")
    
    print("-" * 80)
    print(f"[SUCCESS] Verified {len(records)} records in Supabase public.{table_name}.")
    print("=" * 80)


if __name__ == "__main__":
    verify_uploaded_scenarios()
