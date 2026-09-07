"""
Realistic CI/CD Deployment Telemetry Dataset Generator
Synthesizes 1,500 historical canary deployment records across diverse service architectures.
Features:
- Version, Commit_SHA, Service_Name, Target_Environment, Deployer
- Error_Rate_Delta (%)
- Latency_P95_ms, Latency_P99_ms
- Unit_Test_Failure_Rate (%), Integration_Test_Passed (0/1)
- Security_Critical_CVEs, Security_High_CVEs
- Anomalous_Egress_MB, Auth_Failure_Spike (%)
- Changed_Files_Count, Lines_Delta
- Actual_Production_Incident (0/1), Risk_Score (0-100), Target_Decision (ALLOW / PAUSE / BLOCK)
"""

import os
import sys
import io
import json
import random
import numpy as np
import pandas as pd

# UTF-8 stdout support
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

N_RECORDS = 1500

SERVICES = ["auth-gateway", "payment-service", "order-processing", "user-profile", "notification-worker", "search-indexer", "analytics-pipeline"]
ENVIRONMENTS = ["canary-prod", "staging-preprod", "production-us-east", "production-eu-central"]
DEPLOYERS = ["alex.chen", "sarah.jenkins", "marcus.vance", "elena.rostova", "ci-bot-auto", "david.kim", "priya.sharma"]

records = []

for i in range(N_RECORDS):
    dep_id = f"DEP-{10000 + i}"
    major = random.choice([1, 2, 3])
    minor = random.randint(0, 15)
    patch = random.randint(0, 30)
    version = f"v{major}.{minor}.{patch}"
    sha = f"{random.choice('abcdef0123456789')}{random.choice('abcdef0123456789')}{random.choice('abcdef0123456789')}{random.choice('abcdef0123456789')}{random.choice('abcdef0123456789')}{random.choice('abcdef0123456789')}{random.choice('abcdef0123456789')}"
    
    service = random.choice(SERVICES)
    env = random.choice(ENVIRONMENTS)
    deployer = random.choice(DEPLOYERS)
    
    # 70% normal releases, 18% moderate anomaly releases, 12% critical/high risk releases
    scenario_roll = random.random()
    
    if scenario_roll < 0.70:
        # Healthy Canary
        err_rate_delta = max(0.0, round(float(np.random.normal(0.05, 0.08)), 3))
        p95_lat = max(20.0, round(float(np.random.normal(65.0, 15.0)), 1))
        p99_lat = p95_lat + max(10.0, round(float(np.random.normal(35.0, 12.0)), 1))
        unit_fail = 0.0 if random.random() > 0.15 else round(random.uniform(0.1, 0.5), 2)
        integ_pass = 1
        crit_cves = 0
        high_cves = 0 if random.random() > 0.10 else 1
        egress_mb = round(max(5.0, float(np.random.normal(25.0, 8.0))), 1)
        auth_spike = max(0.0, round(float(np.random.normal(1.2, 1.5)), 1))
        files_cnt = random.randint(1, 15)
        lines_delta = random.randint(10, 450)
        incident = 0
    elif scenario_roll < 0.88:
        # Moderate Anomaly (e.g. latency bump or test flakiness or medium cve)
        err_rate_delta = round(float(np.random.uniform(0.3, 1.8)), 3)
        p95_lat = round(float(np.random.uniform(110.0, 240.0)), 1)
        p99_lat = p95_lat + round(float(np.random.uniform(60.0, 150.0)), 1)
        unit_fail = round(float(np.random.uniform(0.5, 3.5)), 2)
        integ_pass = 1 if random.random() > 0.25 else 0
        crit_cves = 0 if random.random() > 0.20 else 1
        high_cves = random.randint(1, 3)
        egress_mb = round(float(np.random.uniform(40.0, 110.0)), 1)
        auth_spike = round(float(np.random.uniform(5.0, 18.0)), 1)
        files_cnt = random.randint(10, 45)
        lines_delta = random.randint(300, 1800)
        incident = 1 if random.random() > 0.70 else 0
    else:
        # Severe Anomaly / Dangerous Release
        err_rate_delta = round(float(np.random.uniform(2.5, 9.5)), 3)
        p95_lat = round(float(np.random.uniform(280.0, 850.0)), 1)
        p99_lat = p95_lat + round(float(np.random.uniform(200.0, 600.0)), 1)
        unit_fail = round(float(np.random.uniform(4.0, 25.0)), 2)
        integ_pass = 0 if random.random() > 0.20 else 1
        crit_cves = random.randint(1, 4)
        high_cves = random.randint(2, 7)
        egress_mb = round(float(np.random.uniform(150.0, 650.0)), 1)
        auth_spike = round(float(np.random.uniform(25.0, 95.0)), 1)
        files_cnt = random.randint(30, 120)
        lines_delta = random.randint(1500, 8500)
        incident = 1

    # Calibrated risk calculation based on domain signal weightings
    raw_risk = (
        (err_rate_delta * 14.0) +
        (min(p95_lat / 12.0, 25.0)) +
        (unit_fail * 5.5) +
        ((1 - integ_pass) * 20.0) +
        (crit_cves * 25.0) +
        (high_cves * 8.0) +
        (min(auth_spike * 0.4, 15.0)) +
        (min(files_cnt * 0.15, 6.0)) +
        (random.uniform(-3.0, 3.0))
    )
    risk_score = int(np.clip(round(raw_risk), 0, 100))
    
    if risk_score < 50:
        decision = "ALLOW"
    elif risk_score < 75:
        decision = "PAUSE"
    else:
        decision = "BLOCK"
        
    records.append({
        "Deployment_ID": dep_id,
        "Version": version,
        "Commit_SHA": sha,
        "Service_Name": service,
        "Target_Environment": env,
        "Deployer": deployer,
        "Error_Rate_Delta": err_rate_delta,
        "Latency_P95_ms": p95_lat,
        "Latency_P99_ms": p99_lat,
        "Unit_Test_Failure_Rate": unit_fail,
        "Integration_Test_Passed": integ_pass,
        "Security_Critical_CVEs": crit_cves,
        "Security_High_CVEs": high_cves,
        "Anomalous_Egress_MB": egress_mb,
        "Auth_Failure_Spike": auth_spike,
        "Changed_Files_Count": files_cnt,
        "Lines_Delta": lines_delta,
        "Production_Incident": incident,
        "Risk_Score": risk_score,
        "Decision": decision
    })

df = pd.DataFrame(records)
output_path = os.path.join(DATA_DIR, "deployment_telemetry.csv")
df.to_csv(output_path, index=False)
print(f"[+] Successfully generated {len(df)} deployment telemetry records at: {output_path}")
print(f"    Decision Breakdown: {df['Decision'].value_counts().to_dict()}")
print(f"    Mean Risk Score: {df['Risk_Score'].mean():.2f}")
