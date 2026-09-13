import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from stage4_slm.api import app

client = TestClient(app)

print("[TEST] Testing GET /health ...")
resp = client.get("/health")
print("Response code:", resp.status_code)
print("Response json:", resp.json())
assert resp.status_code == 200

print("\n[TEST] Testing POST /summarize ...")
payload = {
    "clinical_report": "Patient diagnosed with EGFR L858R positive NSCLC. Prescribed Erlotinib 150 mg PO daily. Experienced Grade 1 rash, no severe toxicity.",
    "urgency": "Moderate",
    "entities": {
        "drugs": ["Erlotinib"],
        "mutations": ["EGFR L858R"],
        "adverse_events": ["rash"]
    }
}
resp = client.post("/summarize", json=payload)
print("Response code:", resp.status_code)
data = resp.json()
print("Keys in response:", list(data.keys()))
print("Model:", data["model"])
print("Source stage:", data["source_stage"])
print("Safety status:", data["safety_validation"]["status"])
assert resp.status_code == 200
print("\n[SUCCESS] FastAPI endpoint tests passed completely!")
