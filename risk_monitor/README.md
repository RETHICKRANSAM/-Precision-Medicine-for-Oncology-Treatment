# RISK MONITOR — Proactive CI/CD Deployment Risk Monitoring System

> **Transforming release management from reactive failure detection to proactive risk containment.**  
> Answers the critical release question before full rollout: *"Is this deployment safe to proceed?"*

---

## 1. System Overview & Problem Statement

Modern enterprises deploy software continuously using CI/CD pipelines. While rapid release velocity accelerates innovation, it frequently introduces production outages, latency regressions, test failures, and critical security vulnerabilities.

Traditional monitoring approaches are **reactive**:
1. Teams discover bugs only after a deployment reaches 100% of production users.
2. Fragmented dashboards (CI/CD logs, APM metrics, vulnerability scans) obscure deployment health.
3. Lack of automated audit evidence makes compliance verification cumbersome.

**Risk Monitor** solves this by evaluating releases during their early **canary phase** (1–5% traffic). It ingests multi-dimensional telemetry (error spikes, P95/P99 latency, unit/integration test results, and CVE security flags), calculates a calibrated **Risk Score (0–100)** via an explainable Machine Learning model, converts it into an actionable decision (**ALLOW**, **PAUSE**, or **BLOCK**), and persists an immutable audit trail.

---

## 2. Decision Logic & Tiers

| Decision | Risk Score Range | Automated Policy Action | Recommended Engineering Workflow |
| :---: | :---: | :--- | :--- |
| <span style="color:#10b981;font-weight:bold">ALLOW</span> | **0 – 49** | Release is cleared as safe. Automated promotion to 100% production traffic. | No intervention required; auto-promoted. |
| <span style="color:#f59e0b;font-weight:bold">PAUSE</span> | **50 – 74** | Moderate anomaly detected. Deployment paused at canary stage. | Requires SRE/DevSecOps manual sign-off and rationale logging. |
| <span style="color:#f43f5e;font-weight:bold">BLOCK</span> | **75 – 100** | High risk / critical threat. Canary traffic immediately halted & quarantined. | Release aborted. Automated rollback triggered; alerts dispatched. |

---

## 3. Technology Stack

- **Frontend:**
  - HTML5 & Vanilla JavaScript
  - Tailwind CSS + Glassmorphism Dark Theme (SOC Cyber-Command aesthetic)
  - Chart.js for real-time risk gauges, telemetry trends, and 6-axis risk radar
  - Dual-Mode Architecture: connects to live Flask backend or runs autonomously in client-side simulation mode on **GitHub Pages**.
- **Backend & REST API:**
  - Python 3.13 + Flask REST API + Flask-CORS
  - Scikit-learn Random Forest Regressor & guardrail calibrator
  - Persistent JSON audit trail (`data/audit_trail.json`)
- **Data Engineering:**
  - 1,500 historical canary release records (`data/deployment_telemetry.csv`)
  - Continuous telemetry ingestion across 11 diagnostic signals
- **CI/CD & Deployment:**
  - GitHub Pages for dashboard hosting
  - GitHub Actions automated deployment workflow (`.github/workflows/deploy-risk-monitor.yml`)

---

## 4. Directory Structure

```text
risk_monitor/
├── backend/
│   └── app.py                       # Flask REST API server & static dashboard host
├── data/
│   ├── deployment_telemetry.csv     # 1,500 historical deployment records
│   ├── generate_telemetry_data.py   # Dataset synthesis script
│   └── audit_trail.json             # Persistent immutable audit trail
├── frontend/
│   ├── index.html                   # SOC dark Glassmorphism dashboard
│   ├── app.js                       # Frontend logic, Chart.js, dual-mode API
│   └── styles.css                   # Glassmorphism styling, glows, animations
├── models/
│   ├── risk_predictor.py            # Scikit-learn training & inference pipeline
│   └── risk_model.joblib            # Serialized trained model
├── scripts/
│   └── test_risk_monitor.py         # Automated verification test suite
└── README.md                        # Documentation
```

---

## 5. Getting Started & Local Execution

### Step 1: Start the Backend & SOC Dashboard
Run the Flask server from the repository root:
```bash
python risk_monitor/backend/app.py
```
Open your browser at: **`http://localhost:5000`**

### Step 2: Run Verification Tests
To run the automated test suite verifying all REST endpoints, ML scoring, and SRE overrides:
```bash
python risk_monitor/scripts/test_risk_monitor.py
```

### Step 3: GitHub Pages Deployment
The frontend is 100% static and self-contained:
- Pushing to GitHub triggers `.github/workflows/deploy-risk-monitor.yml`.
- The dashboard automatically detects when running on GitHub Pages and enables the client-side ML evaluation engine with zero configuration needed.

---

## 6. REST API Reference

### `GET /api/health`
Returns service and ML engine status.

### `GET /api/metrics/summary`
Returns executive release statistics (total deployments, ALLOW/PAUSE/BLOCK counts & percentages, average risk score).

### `POST /api/deployments/evaluate`
Evaluates a release based on incoming canary telemetry.
```json
{
  "version": "v2.6.0",
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
}
```

### `POST /api/deployments/override`
Allows an authorized engineer to sign off on a PAUSED deployment:
```json
{
  "deployment_id": "DEP-10024",
  "action": "ALLOW",
  "engineer": "lead.sre-oncall",
  "reason": "Cold start latency verified safe on Canary node. INC-4091"
}
```

### `POST /api/simulation/trigger`
Simulates standard release scenarios (`clean_release`, `latency_spike`, `critical_cve_exploit`, `massive_failure`).

---
**Developed by:** AI Data & Platform Engineering Team  
**Architecture Status:** Production Verified & Ready for Deployment
