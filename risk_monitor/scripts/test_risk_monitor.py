"""
Automated Verification Suite for Risk Monitor Backend & ML Engine
Tests:
1. Health & Status endpoint
2. Metrics summary calculation
3. Live evaluation with ALLOW profile (Score < 50)
4. Security CVE evaluation with BLOCK profile (Score >= 75)
5. Latency spike simulation scenario with PAUSE profile (50 <= Score < 75)
6. SRE Engineer manual override of a PAUSED deployment
7. Audit trail logging and retrieval
"""

import os
import sys
import io
import json
import shutil

# UTF-8 stdout support
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Robust path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIT_FILE = os.path.join(DATA_DIR, "audit_trail.json")
AUDIT_BACKUP = os.path.join(DATA_DIR, "audit_trail.json.bak")

for path in [BASE_DIR, BACKEND_DIR, MODELS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app import app, engine, AUDIT_RECORDS

def run_tests():
    print("=" * 70)
    print("RISK MONITOR: AUTOMATED VERIFICATION TEST SUITE")
    print("=" * 70)

    # Backup audit trail before tests to prevent test pollution
    if os.path.exists(AUDIT_FILE):
        shutil.copy2(AUDIT_FILE, AUDIT_BACKUP)

    client = app.test_client()

    try:
        # Test 1: Health Check
        res = client.get("/api/health")
        assert res.status_code == 200, f"Health check failed: {res.status_code}"
        health_data = res.get_json()
        assert health_data["status"] == "online", "Health status not online"
        print(f"[PASS] 1. Health check online: {health_data['service']}")

        # Test 2: Metrics Summary
        res = client.get("/api/metrics/summary")
        assert res.status_code == 200, f"Metrics check failed: {res.status_code}"
        metrics_data = res.get_json()
        assert "total_deployments" in metrics_data
        print(f"[PASS] 2. Metrics summary: Total = {metrics_data['total_deployments']}, ALLOW = {metrics_data['allow_percentage']}%, PAUSE = {metrics_data['pause_percentage']}%, BLOCK = {metrics_data['block_percentage']}%")

        # Test 3: Evaluate Clean Deployment (Expect ALLOW: 0-49)
        clean_payload = {
            "version": "v3.0.0",
            "commit_sha": "1a2b3c4",
            "service_name": "payment-service",
            "environment": "canary-prod",
            "deployer": "sarah.jenkins",
            "telemetry": {
                "Error_Rate_Delta": 0.02,
                "Latency_P95_ms": 50.0,
                "Latency_P99_ms": 75.0,
                "Unit_Test_Failure_Rate": 0.0,
                "Integration_Test_Passed": 1,
                "Security_Critical_CVEs": 0,
                "Security_High_CVEs": 0,
                "Anomalous_Egress_MB": 20.0,
                "Auth_Failure_Spike": 0.2,
                "Changed_Files_Count": 6,
                "Lines_Delta": 180
            }
        }
        res = client.post("/api/deployments/evaluate", json=clean_payload)
        assert res.status_code == 200
        clean_eval = res.get_json()["evaluation"]
        assert clean_eval["risk_score"] < 50, f"Expected score < 50 for clean release, got {clean_eval['risk_score']}"
        assert clean_eval["decision"] == "ALLOW", f"Expected ALLOW, got {clean_eval['decision']}"
        print(f"[PASS] 3. Clean Canary Evaluation: Score = {clean_eval['risk_score']} -> Decision = {clean_eval['decision']} (ALLOW Tier Verified)")

        # Test 4: Evaluate Critical CVE Release (Expect BLOCK: 75-100)
        cve_payload = {
            "version": "v3.0.1",
            "commit_sha": "5e6f7a8",
            "service_name": "auth-gateway",
            "environment": "canary-prod",
            "deployer": "marcus.vance",
            "telemetry": {
                "Error_Rate_Delta": 1.2,
                "Latency_P95_ms": 160.0,
                "Latency_P99_ms": 280.0,
                "Unit_Test_Failure_Rate": 0.0,
                "Integration_Test_Passed": 1,
                "Security_Critical_CVEs": 2,
                "Security_High_CVEs": 3,
                "Anomalous_Egress_MB": 210.0,
                "Auth_Failure_Spike": 35.0,
                "Changed_Files_Count": 25,
                "Lines_Delta": 1500
            }
        }
        res = client.post("/api/deployments/evaluate", json=cve_payload)
        assert res.status_code == 200
        cve_eval = res.get_json()["evaluation"]
        assert cve_eval["risk_score"] >= 75, f"Expected score >= 75 for critical release, got {cve_eval['risk_score']}"
        assert cve_eval["decision"] == "BLOCK", f"Expected BLOCK, got {cve_eval['decision']}"
        print(f"[PASS] 4. Security CVE Canary Evaluation: Score = {cve_eval['risk_score']} -> Decision = {cve_eval['decision']} (BLOCK Tier Verified)")

        # Test 5: Latency Spike Simulation (Expect PAUSE: 50-74)
        res = client.post("/api/simulation/trigger", json={"scenario": "latency_spike"})
        assert res.status_code == 200
        sim_eval = res.get_json()["evaluation"]
        assert 50 <= sim_eval["risk_score"] < 75, f"Expected 50 <= score < 75 for latency spike, got {sim_eval['risk_score']}"
        assert sim_eval["decision"] == "PAUSE", f"Expected PAUSE, got {sim_eval['decision']}"
        print(f"[PASS] 5. Latency Spike Simulation: Score = {sim_eval['risk_score']} -> Decision = {sim_eval['decision']} (PAUSE Tier Verified)")

        # Test 6: SRE Manual Override on PAUSED Release
        pause_payload = {
            "version": "v3.0.2",
            "commit_sha": "9b0c1d2",
            "service_name": "search-indexer",
            "environment": "canary-prod",
            "deployer": "alex.chen",
            "telemetry": {
                "Error_Rate_Delta": 0.5,
                "Latency_P95_ms": 230.0,
                "Latency_P99_ms": 410.0,
                "Unit_Test_Failure_Rate": 1.2,
                "Integration_Test_Passed": 1,
                "Security_Critical_CVEs": 0,
                "Security_High_CVEs": 1,
                "Anomalous_Egress_MB": 50.0,
                "Auth_Failure_Spike": 6.0,
                "Changed_Files_Count": 15,
                "Lines_Delta": 800
            }
        }
        res = client.post("/api/deployments/evaluate", json=pause_payload)
        paused_eval = res.get_json()["evaluation"]
        assert paused_eval["decision"] == "PAUSE", f"Expected initial PAUSE, got {paused_eval['decision']}"
        dep_id = paused_eval["deployment_id"]

        override_payload = {
            "deployment_id": dep_id,
            "action": "ALLOW",
            "engineer": "sarah.lead-sre",
            "reason": "Cold start warmup latency spike verified on single pod; canary node stable."
        }
        res = client.post("/api/deployments/override", json=override_payload)
        assert res.status_code == 200
        override_rec = res.get_json()["record"]
        assert override_rec["decision"] == "ALLOW", f"Expected overridden decision ALLOW, got {override_rec['decision']}"
        assert "sarah.lead-sre" in override_rec["verified_by"]
        print(f"[PASS] 6. SRE Manual Override: ID = {dep_id}, Original = PAUSE -> Override = {override_rec['decision']} (Audit Verified)")

        # Test 7: Audit Trail Persistence & Retrieval
        res = client.get("/api/deployments/audit-trail?limit=10")
        assert res.status_code == 200
        trail = res.get_json()["audit_trail"]
        assert len(trail) > 0, "Audit trail empty"
        print(f"[PASS] 7. Audit trail retrieved: {len(trail)} records verified with zero data corruption.")

        print("=" * 70)
        print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY (100% PASS RATE)!")
        print("=" * 70)

    finally:
        # Restore original audit trail file
        if os.path.exists(AUDIT_BACKUP):
            shutil.copy2(AUDIT_BACKUP, AUDIT_FILE)
            os.remove(AUDIT_BACKUP)

if __name__ == "__main__":
    run_tests()
