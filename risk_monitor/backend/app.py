"""
Risk Monitor - CI/CD Deployment Risk Monitoring & Early Canary Evaluation Platform
Flask REST API Backend & Web Server
"""

import os
import sys
import io
import json
import time
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# UTF-8 stdout support
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
AUDIT_TRAIL_FILE = os.path.join(DATA_DIR, "audit_trail.json")

# Add models directory to path
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

from risk_predictor import RiskInferenceEngine

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Initialize ML inference engine
engine = RiskInferenceEngine()

def load_audit_trail():
    if os.path.exists(AUDIT_TRAIL_FILE):
        try:
            with open(AUDIT_TRAIL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    # If file doesn't exist, bootstrap from deployment_telemetry.csv
    telemetry_csv = os.path.join(DATA_DIR, "deployment_telemetry.csv")
    initial_records = []
    if os.path.exists(telemetry_csv):
        import pandas as pd
        df = pd.read_csv(telemetry_csv)
        sample = df.tail(25)
        
        base_time = datetime.datetime.now() - datetime.timedelta(hours=48)
        for idx, (_, r) in enumerate(sample.iterrows()):
            eval_time = base_time + datetime.timedelta(minutes=idx * 110)
            score = int(r["Risk_Score"])
            dec = r["Decision"]
            color = "emerald" if dec == "ALLOW" else ("amber" if dec == "PAUSE" else "rose")
            
            reasons = []
            if float(r["Security_Critical_CVEs"]) > 0:
                reasons.append(f"{int(r['Security_Critical_CVEs'])} critical CVE security vulnerability detected.")
            if float(r["Error_Rate_Delta"]) > 1.0:
                reasons.append(f"Canary error rate spiked +{r['Error_Rate_Delta']:.2f}%.")
            if float(r["Latency_P95_ms"]) > 180:
                reasons.append(f"Latency P95 elevated at {r['Latency_P95_ms']}ms.")
            if not reasons:
                reasons.append("All canary metrics within strict operational baselines.")
                
            initial_records.append({
                "deployment_id": r["Deployment_ID"],
                "timestamp": eval_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "version": r["Version"],
                "commit_sha": r["Commit_SHA"][:7],
                "service_name": r["Service_Name"],
                "environment": r["Target_Environment"],
                "deployer": r["Deployer"],
                "risk_score": score,
                "decision": dec,
                "color": color,
                "status_text": "Approved & Deployed" if dec == "ALLOW" else ("Awaiting Verification" if dec == "PAUSE" else "Halted & Quarantined"),
                "key_factors": reasons,
                "verified_by": "auto-policy-engine" if dec != "PAUSE" else "pending",
                "telemetry": {
                    "Error_Rate_Delta": float(r["Error_Rate_Delta"]),
                    "Latency_P95_ms": float(r["Latency_P95_ms"]),
                    "Unit_Test_Failure_Rate": float(r["Unit_Test_Failure_Rate"]),
                    "Security_Critical_CVEs": int(r["Security_Critical_CVEs"])
                }
            })
            
    save_audit_trail(initial_records)
    return initial_records

def save_audit_trail(records):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(AUDIT_TRAIL_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

# Load existing audit records
AUDIT_RECORDS = load_audit_trail()

# -------------------------------------------------------------
# REST API ENDPOINTS
# -------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Risk Monitor Proactive Deployment Security Engine",
        "ml_engine": "RandomForestRegressor + Guardrail Calibrator",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })

@app.route("/api/metrics/summary", methods=["GET"])
def metrics_summary():
    total = len(AUDIT_RECORDS)
    if total == 0:
        return jsonify({"total": 0, "allow": 0, "pause": 0, "block": 0, "avg_score": 0})
        
    allow_cnt = sum(1 for r in AUDIT_RECORDS if r["decision"] == "ALLOW")
    pause_cnt = sum(1 for r in AUDIT_RECORDS if r["decision"] == "PAUSE")
    block_cnt = sum(1 for r in AUDIT_RECORDS if r["decision"] == "BLOCK")
    avg_score = round(sum(r["risk_score"] for r in AUDIT_RECORDS) / total, 1)
    
    return jsonify({
        "total_deployments": total,
        "allow_count": allow_cnt,
        "allow_percentage": round((allow_cnt / total) * 100, 1),
        "pause_count": pause_cnt,
        "pause_percentage": round((pause_cnt / total) * 100, 1),
        "block_count": block_cnt,
        "block_percentage": round((block_cnt / total) * 100, 1),
        "average_risk_score": avg_score
    })

@app.route("/api/deployments/audit-trail", methods=["GET"])
def get_audit_trail():
    limit = request.args.get("limit", default=50, type=int)
    # Return sorted by newest first
    sorted_records = list(reversed(AUDIT_RECORDS))[:limit]
    return jsonify({
        "count": len(sorted_records),
        "audit_trail": sorted_records
    })

@app.route("/api/deployments/evaluate", methods=["POST"])
def evaluate_deployment():
    data = request.get_json() or {}
    
    version = data.get("version", f"v2.{int(time.time())%100}.0")
    commit_sha = data.get("commit_sha", "a7f29b1")[:7]
    service = data.get("service_name", "order-processing")
    environment = data.get("environment", "canary-prod")
    deployer = data.get("deployer", "ci-bot-auto")
    
    telemetry = data.get("telemetry", {})
    
    # Run Scikit-learn inference engine
    eval_result = engine.evaluate_telemetry(telemetry)
    
    audit_entry = {
        "deployment_id": f"DEP-{len(AUDIT_RECORDS) + 10001}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "version": version,
        "commit_sha": commit_sha,
        "service_name": service,
        "environment": environment,
        "deployer": deployer,
        "risk_score": eval_result["risk_score"],
        "decision": eval_result["decision"],
        "color": eval_result["color"],
        "status_text": eval_result["status_text"],
        "key_factors": eval_result["key_factors"],
        "verified_by": "auto-policy-engine" if eval_result["decision"] != "PAUSE" else "pending",
        "telemetry": telemetry,
        "radar_signals": eval_result["radar_signals"]
    }
    
    AUDIT_RECORDS.append(audit_entry)
    save_audit_trail(AUDIT_RECORDS)
    
    return jsonify({
        "status": "success",
        "evaluation": audit_entry
    })

@app.route("/api/deployments/override", methods=["POST"])
def override_deployment():
    data = request.get_json() or {}
    deployment_id = data.get("deployment_id")
    override_action = data.get("action", "ALLOW") # ALLOW or BLOCK
    reason = data.get("reason", "Signed off by SRE On-call engineer.")
    engineer = data.get("engineer", "sre-lead")
    
    for r in AUDIT_RECORDS:
        if r["deployment_id"] == deployment_id:
            r["decision"] = override_action
            r["color"] = "emerald" if override_action == "ALLOW" else "rose"
            r["status_text"] = f"Manual Override ({override_action}): {reason}"
            r["verified_by"] = f"{engineer} (Override)"
            r["key_factors"].append(f"Manual Override by {engineer}: {reason}")
            save_audit_trail(AUDIT_RECORDS)
            return jsonify({"status": "success", "record": r})
            
    return jsonify({"status": "error", "message": "Deployment ID not found"}), 404

@app.route("/api/simulation/trigger", methods=["POST"])
def trigger_simulation():
    data = request.get_json() or {}
    scenario = data.get("scenario", "clean_release")
    
    scenarios = {
        "clean_release": {
            "version": "v2.5.0-rc1",
            "commit_sha": "4f89ac2",
            "service_name": "payment-service",
            "environment": "canary-prod",
            "deployer": "sarah.jenkins",
            "telemetry": {
                "Error_Rate_Delta": 0.04,
                "Latency_P95_ms": 58.0,
                "Latency_P99_ms": 82.0,
                "Unit_Test_Failure_Rate": 0.0,
                "Integration_Test_Passed": 1,
                "Security_Critical_CVEs": 0,
                "Security_High_CVEs": 0,
                "Anomalous_Egress_MB": 22.0,
                "Auth_Failure_Spike": 0.5,
                "Changed_Files_Count": 8,
                "Lines_Delta": 240
            }
        },
        "latency_spike": {
            "version": "v2.5.0-rc2",
            "commit_sha": "e29ba71",
            "service_name": "search-indexer",
            "environment": "canary-prod",
            "deployer": "alex.chen",
            "telemetry": {
                "Error_Rate_Delta": 0.45,
                "Latency_P95_ms": 230.0,
                "Latency_P99_ms": 420.0,
                "Unit_Test_Failure_Rate": 1.2,
                "Integration_Test_Passed": 1,
                "Security_Critical_CVEs": 0,
                "Security_High_CVEs": 1,
                "Anomalous_Egress_MB": 65.0,
                "Auth_Failure_Spike": 7.5,
                "Changed_Files_Count": 24,
                "Lines_Delta": 1150
            }
        },
        "critical_cve_exploit": {
            "version": "v2.5.0-rc3",
            "commit_sha": "9a01f4c",
            "service_name": "auth-gateway",
            "environment": "canary-prod",
            "deployer": "marcus.vance",
            "telemetry": {
                "Error_Rate_Delta": 1.85,
                "Latency_P95_ms": 190.0,
                "Latency_P99_ms": 310.0,
                "Unit_Test_Failure_Rate": 0.0,
                "Integration_Test_Passed": 1,
                "Security_Critical_CVEs": 2,
                "Security_High_CVEs": 4,
                "Anomalous_Egress_MB": 240.0,
                "Auth_Failure_Spike": 45.0,
                "Changed_Files_Count": 42,
                "Lines_Delta": 3100
            }
        },
        "massive_failure": {
            "version": "v2.5.0-rc4",
            "commit_sha": "d3810ae",
            "service_name": "order-processing",
            "environment": "canary-prod",
            "deployer": "ci-bot-auto",
            "telemetry": {
                "Error_Rate_Delta": 5.40,
                "Latency_P95_ms": 520.0,
                "Latency_P99_ms": 890.0,
                "Unit_Test_Failure_Rate": 14.5,
                "Integration_Test_Passed": 0,
                "Security_Critical_CVEs": 1,
                "Security_High_CVEs": 3,
                "Anomalous_Egress_MB": 380.0,
                "Auth_Failure_Spike": 60.0,
                "Changed_Files_Count": 68,
                "Lines_Delta": 5200
            }
        }
    }
    
    sim_data = scenarios.get(scenario, scenarios["clean_release"])
    eval_result = engine.evaluate_telemetry(sim_data["telemetry"])
    
    audit_entry = {
        "deployment_id": f"DEP-{len(AUDIT_RECORDS) + 10001}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "version": sim_data["version"],
        "commit_sha": sim_data["commit_sha"],
        "service_name": sim_data["service_name"],
        "environment": sim_data["environment"],
        "deployer": sim_data["deployer"],
        "risk_score": eval_result["risk_score"],
        "decision": eval_result["decision"],
        "color": eval_result["color"],
        "status_text": eval_result["status_text"],
        "key_factors": eval_result["key_factors"],
        "verified_by": "auto-policy-engine" if eval_result["decision"] != "PAUSE" else "pending",
        "telemetry": sim_data["telemetry"],
        "radar_signals": eval_result["radar_signals"]
    }
    
    AUDIT_RECORDS.append(audit_entry)
    save_audit_trail(AUDIT_RECORDS)
    
    return jsonify({
        "status": "success",
        "scenario": scenario,
        "evaluation": audit_entry
    })

# -------------------------------------------------------------
# STATIC FRONTEND SERVING
# -------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Risk Monitor REST API & SOC Dashboard starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
