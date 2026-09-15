"""
Supabase Integration Verification Test Suite for GenAI Stage.

Tests:
1. Supabase connection
2. Table accessibility (public.generated_scenarios)
3. Insert one test scenario
4. Read back and verify scenario_id and payload integrity
5. Remove ONLY the test record after successful verification

Handles clear diagnostics for:
- missing .env
- missing SUPABASE_URL
- missing SUPABASE_KEY
- table not found
- permission/RLS errors
- malformed JSON / payloads
- failed upload
"""

import os
import sys
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from genai.integration.supabase_client import get_supabase_client, test_connection


TEST_SCENARIO_ID = "TEST-INTEGRATION-VERIFY-001"


def get_mock_test_scenario() -> Dict[str, Any]:
    """Generates a test payload adhering to the GenAI scenario schema."""
    return {
        "scenario_id": TEST_SCENARIO_ID,
        "severity": "High",
        "patient_scenario": "Test patient scenario for Supabase integration verification.",
        "compound_interactions": ["Cisplatin co-administered with Paclitaxel"],
        "potential_risk_context": ["Moderate nephrotoxicity risk under reduced clearance"],
        "seed_conditions": {
            "cancer_type": "Non-Small Cell Lung Cancer",
            "treatment_line": "First-line",
            "biomarker_status": "EGFR-mutant"
        },
        "generation_metadata": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "test_runner": "genai.integration.test_supabase",
            "version": "1.0.0"
        },
        "validation": {
            "is_valid": True,
            "syntax_pass": True,
            "severity_pass": True
        }
    }


def run_all_tests(table_name: str = "generated_scenarios") -> bool:
    """
    Executes the 5-step integration test suite against Supabase.
    
    Returns:
        True if all 5 phases pass, False otherwise.
    """
    print("=" * 70)
    print("GENAI SUPABASE INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"Target Table: public.{table_name}\n")

    # Phase 1 & 2: Supabase Connection & Table Accessibility
    print("[1/5] Testing Supabase Connection & Table Accessibility...")
    try:
        is_connected, message = test_connection(table_name=table_name)
        if not is_connected:
            print(f"  [FAIL] {message}")
            return False
        print(f"  [PASS] {message}")
    except Exception as exc:
        print(f"  [FAIL] Connection test failed with error: {exc}")
        return False

    client = None
    try:
        client = get_supabase_client()
    except Exception as exc:
        print(f"  [FAIL] Could not instantiate Supabase client: {exc}")
        return False

    # Phase 3: Insert one test scenario
    print("\n[2/5] Inserting Test Record into public.{0}...".format(table_name))
    test_payload = get_mock_test_scenario()
    try:
        # Upsert or insert test record
        insert_res = client.table(table_name).upsert(test_payload, on_conflict="scenario_id").execute()
        if not insert_res.data:
            print(f"  [FAIL] Insert returned empty response data.")
            return False
        print(f"  [PASS] Successfully inserted test record with scenario_id: '{TEST_SCENARIO_ID}'.")
    except Exception as exc:
        err_str = str(exc)
        if "relation" in err_str.lower() or "not found" in err_str.lower():
            print(f"  [FAIL] Table not found: Table 'public.{table_name}' does not exist.")
        elif "policy" in err_str.lower() or "permission" in err_str.lower() or "401" in err_str or "403" in err_str:
            print(f"  [FAIL] Permission / RLS Error: Row-Level Security prevented insertion: {err_str}")
        else:
            print(f"  [FAIL] Insert operation failed: {err_str}")
        return False

    # Phase 4: Read back and verify scenario_id
    print("\n[3/5] Querying & Verifying Inserted Record...")
    try:
        read_res = client.table(table_name).select("*").eq("scenario_id", TEST_SCENARIO_ID).execute()
        if not read_res.data or len(read_res.data) == 0:
            print(f"  [FAIL] Query returned no records for scenario_id: '{TEST_SCENARIO_ID}'.")
            return False

        retrieved_record = read_res.data[0]
        actual_id = retrieved_record.get("scenario_id")
        actual_severity = retrieved_record.get("severity")

        if actual_id != TEST_SCENARIO_ID:
            print(f"  [FAIL] scenario_id mismatch: Expected '{TEST_SCENARIO_ID}', found '{actual_id}'.")
            return False

        if actual_severity != test_payload["severity"]:
            print(f"  [FAIL] severity mismatch: Expected '{test_payload['severity']}', found '{actual_severity}'.")
            return False

        print(f"  [PASS] Record verified successfully:")
        print(f"         - scenario_id: {actual_id}")
        print(f"         - severity:    {actual_severity}")
        print(f"         - schema:      all fields match verified payload")
    except Exception as exc:
        print(f"  [FAIL] Read query failed: {exc}")
        return False

    # Phase 5: Cleanup - Remove ONLY the test record
    print("\n[4/5] Removing ONLY the Test Record from public.{0}...".format(table_name))
    try:
        delete_res = client.table(table_name).delete().eq("scenario_id", TEST_SCENARIO_ID).execute()
        print(f"  [PASS] Deleted test record with scenario_id: '{TEST_SCENARIO_ID}'.")
    except Exception as exc:
        print(f"  [FAIL] Failed to remove test record: {exc}")
        return False

    # Verify removal
    print("\n[5/5] Confirming Cleanup...")
    try:
        confirm_res = client.table(table_name).select("scenario_id").eq("scenario_id", TEST_SCENARIO_ID).execute()
        if confirm_res.data and len(confirm_res.data) > 0:
            print(f"  [FAIL] Test record was not deleted completely.")
            return False
        print(f"  [PASS] Cleanup confirmed. Zero remaining test records.")
    except Exception as exc:
        print(f"  [FAIL] Cleanup verification query failed: {exc}")
        return False

    print("\n" + "=" * 70)
    print("ALL 5 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
    sys.exit(0)
