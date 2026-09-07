"""
Machine Learning Risk Prediction Pipeline for CI/CD Deployments
Trains a calibrated Random Forest regressor on deployment telemetry signals.
Provides an inference engine with explainable feature importance attribution.
"""

import os
import sys
import io
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# UTF-8 stdout support
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "deployment_telemetry.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Error_Rate_Delta",
    "Latency_P95_ms",
    "Latency_P99_ms",
    "Unit_Test_Failure_Rate",
    "Integration_Test_Passed",
    "Security_Critical_CVEs",
    "Security_High_CVEs",
    "Anomalous_Egress_MB",
    "Auth_Failure_Spike",
    "Changed_Files_Count",
    "Lines_Delta"
]

FEATURE_LABELS = {
    "Error_Rate_Delta": "Error Rate Delta (%)",
    "Latency_P95_ms": "P95 Latency (ms)",
    "Latency_P99_ms": "P99 Latency (ms)",
    "Unit_Test_Failure_Rate": "Unit Test Failures (%)",
    "Integration_Test_Passed": "Integration Test Status",
    "Security_Critical_CVEs": "Critical CVE Flags",
    "Security_High_CVEs": "High CVE Flags",
    "Anomalous_Egress_MB": "Anomalous Network Egress (MB)",
    "Auth_Failure_Spike": "Auth Failure Spikes (%)",
    "Changed_Files_Count": "Changed Files Count",
    "Lines_Delta": "Lines Changed Delta"
}

def train_and_save_model():
    print("[*] Loading deployment telemetry training data...")
    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURE_COLS]
    y = df["Risk_Score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print(f"[*] Training dataset: {X_train.shape[0]} samples | Test dataset: {X_test.shape[0]} samples")
    
    regressor = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[+] Model Evaluation: MSE = {mse:.2f}, R2 Score = {r2:.4f}")
    
    # Feature importances
    importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in regressor.feature_importances_]))
    print("[+] Top Predictive Risk Signals:")
    for f, imp in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"    - {FEATURE_LABELS[f]}: {imp*100:.1f}%")
        
    bundle = {
        "model": regressor,
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "importances": importances,
        "metrics": {"mse": mse, "r2": r2}
    }
    
    joblib.dump(bundle, MODEL_PATH)
    print(f"[+] Serialized model package to: {MODEL_PATH}")
    return bundle

class RiskInferenceEngine:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            train_and_save_model()
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.feature_cols = self.bundle["feature_cols"]
        self.feature_labels = self.bundle["feature_labels"]
        self.importances = self.bundle["importances"]

    def evaluate_telemetry(self, telemetry_dict):
        """
        Takes raw telemetry input dictionary and outputs:
        - risk_score (0 - 100)
        - decision (ALLOW / PAUSE / BLOCK)
        - decision_color (emerald / amber / rose)
        - explanation (primary factors driving the score)
        - factor_breakdown
        """
        row = []
        for col in self.feature_cols:
            val = telemetry_dict.get(col, 0)
            row.append(float(val))
            
        x_input = pd.DataFrame([row], columns=self.feature_cols)
        predicted_raw = float(self.model.predict(x_input)[0])
        
        # Guardrail overrides (Security & SRE rules)
        # Any Critical CVE automatically pushes into at least PAUSE, >= 2 pushes into BLOCK
        crit_cves = float(telemetry_dict.get("Security_Critical_CVEs", 0))
        integ_passed = float(telemetry_dict.get("Integration_Test_Passed", 1))
        err_delta = float(telemetry_dict.get("Error_Rate_Delta", 0))
        p95 = float(telemetry_dict.get("Latency_P95_ms", 0))
        p99 = float(telemetry_dict.get("Latency_P99_ms", 0))
        unit_fail = float(telemetry_dict.get("Unit_Test_Failure_Rate", 0))
        
        if crit_cves >= 2:
            predicted_raw = max(predicted_raw, 85.0)
        elif crit_cves == 1:
            predicted_raw = max(predicted_raw, 62.0)
            
        if integ_passed == 0:
            predicted_raw = max(predicted_raw, 75.0)
            
        if err_delta >= 4.0:
            predicted_raw = max(predicted_raw, 88.0)
        elif err_delta >= 1.5:
            predicted_raw = max(predicted_raw, 60.0)

        if p95 >= 200.0 or p99 >= 380.0:
            predicted_raw = max(predicted_raw, 58.0)
            
        if unit_fail >= 10.0:
            predicted_raw = max(predicted_raw, 78.0)
        elif unit_fail >= 1.0:
            predicted_raw = max(predicted_raw, 52.0)

        risk_score = int(np.clip(round(predicted_raw), 0, 100))
        
        # Decision mapping
        if risk_score < 50:
            decision = "ALLOW"
            status_text = "Deployment Cleared for Production"
            color = "emerald"
            badge = "ALLOW (Safe)"
        elif risk_score < 75:
            decision = "PAUSE"
            status_text = "Manual Engineer Verification Required"
            color = "amber"
            badge = "PAUSE (Moderate Risk)"
        else:
            decision = "BLOCK"
            status_text = "Release Halted - Unsafe Telemetry Detected"
            color = "rose"
            badge = "BLOCK (Critical Risk)"
            
        # Explanatory factor breakdown
        factors = []
        if crit_cves > 0:
            factors.append(f"{int(crit_cves)} Critical Security CVE vulnerability flag(s) present.")
        if err_delta > 1.0:
            factors.append(f"Elevated canary error rate delta: +{err_delta:.2f}%.")
        if float(telemetry_dict.get('Latency_P95_ms', 0)) > 200:
            factors.append(f"High P95 latency: {telemetry_dict.get('Latency_P95_ms')}ms exceeding 200ms threshold.")
        if float(telemetry_dict.get('Unit_Test_Failure_Rate', 0)) > 2.0:
            factors.append(f"Unit test failure rate at {telemetry_dict.get('Unit_Test_Failure_Rate')}%.")
        if integ_passed == 0:
            factors.append("Integration test suite failed during canary execution.")
        if float(telemetry_dict.get('Auth_Failure_Spike', 0)) > 10.0:
            factors.append(f"Anomalous auth failure spike: +{telemetry_dict.get('Auth_Failure_Spike')}%.")
        if float(telemetry_dict.get('Anomalous_Egress_MB', 0)) > 100.0:
            factors.append(f"Suspicious network egress bandwidth: {telemetry_dict.get('Anomalous_Egress_MB')} MB.")
            
        if not factors:
            factors.append("All canary telemetry within optimal operational baselines.")
            
        return {
            "risk_score": risk_score,
            "decision": decision,
            "status_text": status_text,
            "badge": badge,
            "color": color,
            "key_factors": factors,
            "radar_signals": {
                "Error Rate": min(100, int(float(telemetry_dict.get('Error_Rate_Delta', 0)) * 15)),
                "Latency": min(100, int(float(telemetry_dict.get('Latency_P95_ms', 50)) / 4.0)),
                "Test Failures": min(100, int(float(telemetry_dict.get('Unit_Test_Failure_Rate', 0)) * 8 + (1 - integ_passed) * 40)),
                "Vulnerabilities": min(100, int(crit_cves * 35 + float(telemetry_dict.get('Security_High_CVEs', 0)) * 12)),
                "Network / Auth": min(100, int(float(telemetry_dict.get('Auth_Failure_Spike', 0)) * 2 + float(telemetry_dict.get('Anomalous_Egress_MB', 20)) / 5.0)),
                "Code Blast Radius": min(100, int(float(telemetry_dict.get('Changed_Files_Count', 5)) * 1.5 + float(telemetry_dict.get('Lines_Delta', 100)) / 50.0))
            }
        }

if __name__ == "__main__":
    train_and_save_model()
    
    # Test evaluation
    engine = RiskInferenceEngine()
    test_sample = {
        "Error_Rate_Delta": 3.8,
        "Latency_P95_ms": 380,
        "Latency_P99_ms": 520,
        "Unit_Test_Failure_Rate": 5.2,
        "Integration_Test_Passed": 0,
        "Security_Critical_CVEs": 2,
        "Security_High_CVEs": 3,
        "Anomalous_Egress_MB": 180,
        "Auth_Failure_Spike": 32,
        "Changed_Files_Count": 42,
        "Lines_Delta": 2400
    }
    result = engine.evaluate_telemetry(test_sample)
    print("\n[+] Verification Test Output:")
    print(f"    Score:    {result['risk_score']} / 100")
    print(f"    Decision: {result['decision']}")
    print(f"    Factors:  {result['key_factors']}")
