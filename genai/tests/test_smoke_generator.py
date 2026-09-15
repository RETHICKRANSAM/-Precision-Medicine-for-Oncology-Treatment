"""
Smoke Test Suite for the GenAI SLM-Based Compound Scenario Generator.

Validates the 11 critical operational specifications from Task 12:
1. Mild scenario
2. Moderate scenario
3. Severe scenario
4. Wildcard scenario
5. Multiple-condition interaction
6. Additional seed fields if available
7. Missing seed condition
8. Invalid severity
9. Empty SLM response
10. Invalid generated output
11. Contradictory generated scenario
"""

import pytest
from genai.generation.prompt_builder import PromptBuilder
from genai.generation.scenario_validator import ScenarioValidator
from genai.generation.scenario_generator import CompoundScenarioGenerator


@pytest.fixture(scope="module")
def generator():
    """Initializes a shared generator instance for the test suite."""
    return CompoundScenarioGenerator(device="cpu")


@pytest.fixture
def validator():
    """Returns a fresh validator instance."""
    return ScenarioValidator()


@pytest.fixture
def prompt_builder():
    """Returns a fresh prompt builder instance."""
    return PromptBuilder()


# ---------------------------------------------------------------------------
# Test 1: Mild Scenario
# ---------------------------------------------------------------------------
def test_smoke_01_mild_scenario(generator):
    seeds = {
        "Tumor Mutation Burden": "Low (2.1 mut/Mb)",
        "ctDNA Trend": "Stable / Indolent",
        "Renal Function": "Normal (Creatinine 0.9 mg/dL)",
        "Organ Involvement": "Localized Lung (Stage I)"
    }
    result = generator.generate_scenario(seeds, severity="Mild", scenario_id="TEST-01-MILD")
    assert result["severity"] == "Mild"
    assert result["validation"]["is_valid"] is True
    assert len(result["patient_scenario"]) > 40
    assert len(result["compound_interactions"]) >= 1
    assert "Mild" in result["severity"]


# ---------------------------------------------------------------------------
# Test 2: Moderate Scenario
# ---------------------------------------------------------------------------
def test_smoke_02_moderate_scenario(generator):
    seeds = {
        "Tumor Mutation Burden": "Intermediate (7.4 mut/Mb)",
        "ctDNA Trend": "Moderate fluctuation (3.2 ng/mL)",
        "Renal Function": "Borderline clearance (Creatinine 1.3 mg/dL)",
        "Organ Involvement": "Locally advanced NSCLC"
    }
    result = generator.generate_scenario(seeds, severity="Moderate", scenario_id="TEST-02-MOD")
    assert result["severity"] == "Moderate"
    assert result["validation"]["is_valid"] is True
    assert len(result["potential_risk_context"]) >= 1


# ---------------------------------------------------------------------------
# Test 3: Severe Scenario
# ---------------------------------------------------------------------------
def test_smoke_03_severe_scenario(generator):
    seeds = {
        "Tumor Mutation Burden": "High (24.8 mut/Mb)",
        "ctDNA Trend": "Exponential surge (+85% in 14 days)",
        "Renal Function": "Acute impairment (Creatinine 2.1 mg/dL)",
        "Organ Involvement": "Multi-organ Metastatic (Liver, Bone, Lung)"
    }
    result = generator.generate_scenario(seeds, severity="Severe", scenario_id="TEST-03-SEV")
    assert result["severity"] == "Severe"
    assert result["validation"]["is_valid"] is True
    narrative_lower = result["patient_scenario"].lower()
    assert any(term in narrative_lower for term in ["severe", "high", "acute", "exponential", "metastatic", "crisis", "urgency", "impairment"])


# ---------------------------------------------------------------------------
# Test 4: Wildcard Scenario
# ---------------------------------------------------------------------------
def test_smoke_04_wildcard_scenario(generator):
    seeds = {
        "Tumor Mutation Burden": "Ultra-High (45 mut/Mb)",
        "ctDNA Trend": "Discordant clearance (undetectable in plasma)",
        "Renal Function": "Creatinine 1.1 mg/dL",
        "Organ Involvement": "Colorectal Carcinoma with peritoneal seeding"
    }
    result = generator.generate_scenario(seeds, severity="Wildcard", scenario_id="TEST-04-WILD")
    assert result["severity"] == "Wildcard"
    assert result["validation"]["is_valid"] is True
    narrative_lower = result["patient_scenario"].lower()
    assert any(w in narrative_lower for w in ["atypical", "discordant", "unusual", "wildcard", "peritoneal", "colorectal", "progressive", "idiosyncratic"])


# ---------------------------------------------------------------------------
# Test 5: Multiple-Condition Interaction Verification
# ---------------------------------------------------------------------------
def test_smoke_05_multiple_condition_interaction(generator):
    seeds = {
        "Tumor Mutation Burden": "High TMB",
        "ctDNA Trend": "Rising ctDNA",
        "Renal Function": "Elevated Creatinine 1.9 mg/dL",
        "Organ Involvement": "Renal cell carcinoma and pulmonary nodules"
    }
    result = generator.generate_scenario(seeds, severity="Severe", scenario_id="TEST-05-INTERACTION")
    interactions = result.get("compound_interactions", [])
    assert len(interactions) >= 2
    # Verify interaction discusses relationships between conditions, not just a bare copy
    interaction_text = " ".join(interactions).lower()
    assert any(term in interaction_text for term in [
        "correlat", "synerg", "combin", "impair", "clearance", "stress",
        "rate", "dynamic", "turnover", "lead", "metastat", "frequent", "proliferation"
    ])


# ---------------------------------------------------------------------------
# Test 6: Additional Seed Fields (Extended Clinical Schema)
# ---------------------------------------------------------------------------
def test_smoke_06_additional_seed_fields(generator):
    seeds = {
        "Tumor Mutation Burden": "12.0 mut/Mb",
        "ctDNA Trend": "Rising",
        "Renal Function": "Creatinine 1.2 mg/dL",
        "Organ Involvement": "NSCLC",
        # Extended fields from master clinical datasets
        "Cancer Stage": "Stage IV",
        "Genomic Mutation": "EGFR L858R",
        "Current Antineoplastic Drug": "Osimertinib",
        "Drug Dosage (mg)": "80 mg daily",
        "Reported Symptoms": "Persistent cough and mild fatigue",
        "Total Bilirubin (Hepatic)": "1.4 mg/dL",
        "Baseline Toxicity Risk": "Moderate"
    }
    result = generator.generate_scenario(seeds, severity="Moderate", scenario_id="TEST-06-EXTENDED")
    assert result["validation"]["is_valid"] is True
    # Verify all extended seeds are preserved in seed_conditions dictionary
    for k in seeds:
        assert k in result["seed_conditions"]


# ---------------------------------------------------------------------------
# Test 7: Missing Seed Condition Handling
# ---------------------------------------------------------------------------
def test_smoke_07_missing_seed_condition(generator):
    empty_seeds = {}
    result = generator.generate_scenario(empty_seeds, severity="Mild", scenario_id="TEST-07-EMPTY")
    assert result["validation"]["status"] == "failed"
    assert result["validation"]["is_valid"] is False
    assert any("empty" in e.lower() or "missing" in e.lower() for e in result["validation"]["errors"])


# ---------------------------------------------------------------------------
# Test 8: Invalid Severity Handling
# ---------------------------------------------------------------------------
def test_smoke_08_invalid_severity(generator):
    seeds = {
        "Tumor Mutation Burden": "Low",
        "ctDNA Trend": "Stable",
        "Renal Function": "Creatinine 1.0",
        "Organ Involvement": "Breast"
    }
    result = generator.generate_scenario(seeds, severity="ExtremeDisaster", scenario_id="TEST-08-INV-SEV")
    assert result["validation"]["is_valid"] is False
    assert any("severity" in e.lower() for e in result["validation"]["errors"])


# ---------------------------------------------------------------------------
# Test 9: Empty SLM Response Handling
# ---------------------------------------------------------------------------
def test_smoke_09_empty_slm_response(validator):
    val = validator.validate_scenario("", expected_seeds={"ctDNA": "Low"}, expected_severity="Mild")
    assert val["is_valid"] is False
    assert val["status"] == "failed"
    assert any("empty" in e.lower() for e in val["errors"])


# ---------------------------------------------------------------------------
# Test 10: Invalid Structured Output Handling
# ---------------------------------------------------------------------------
def test_smoke_10_invalid_structured_output(validator):
    malformed = {
        "scenario_id": "TEST-10",
        "severity": "Mild",
        # Missing 'patient_scenario', 'compound_interactions', 'potential_risk_context'
    }
    val = validator.validate_scenario(malformed, expected_seeds={"TMB": "5"}, expected_severity="Mild")
    assert val["is_valid"] is False
    assert any("missing required" in e.lower() for e in val["errors"])


# ---------------------------------------------------------------------------
# Test 11: Contradictory Generated Scenario Detection
# ---------------------------------------------------------------------------
def test_smoke_11_contradictory_scenario(validator):
    # Seed indicates severe renal impairment (Creatinine 2.4 mg/dL)
    seeds = {"Renal Function": "Severe failure with Creatinine 2.4 mg/dL"}
    contradictory_scenario = {
        "scenario_id": "TEST-11",
        "seed_conditions": seeds,
        "severity": "Severe",
        # Scenario falsely asserts completely normal renal function
        "patient_scenario": "The patient demonstrates completely normal renal function with unremarkable kidney clearance.",
        "compound_interactions": ["Normal renal function allows standard high-dose therapy."],
        "potential_risk_context": ["No renal risk observed."]
    }
    val = validator.validate_scenario(contradictory_scenario, expected_seeds=seeds, expected_severity="Severe")
    assert val["is_valid"] is False
    assert any("contradiction" in e.lower() for e in val["errors"])
