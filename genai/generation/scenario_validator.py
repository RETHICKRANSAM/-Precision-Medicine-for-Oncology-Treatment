"""
Scenario Validator for the GenAI SLM-Based Compound Scenario Generator.

Validates generated scenarios against:
- Missing seed conditions
- Invalid seed values
- Invalid severity
- Empty SLM response
- Invalid structured output schema
- Omission of critical seed conditions
- Contradictions with seed values
- Unsupported claims and invented measurements
- Duplicate scenarios across batches
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Tuple, Set, Optional

VALID_SEVERITIES = {"Mild", "Moderate", "Severe", "Wildcard"}
REQUIRED_FIELDS = [
    "scenario_id",
    "seed_conditions",
    "severity",
    "patient_scenario",
    "compound_interactions",
    "potential_risk_context"
]


class ScenarioValidator:
    """Multi-tier validator and quality auditor for generated compound scenarios."""

    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def reset_duplicate_cache(self):
        """Clears the internal duplicate scenario cache."""
        self.seen_hashes.clear()

    def validate_seed_input(self, seed_conditions: Any, severity: str) -> Tuple[bool, List[str]]:
        """Validates input seed conditions and requested severity before generation."""
        errors: List[str] = []

        if not seed_conditions or not isinstance(seed_conditions, dict):
            errors.append("Missing or invalid seed_conditions: must be a non-empty dictionary.")
            return False, errors

        if len(seed_conditions) == 0:
            errors.append("Seed conditions dictionary is empty.")

        # Severity check
        sev_title = str(severity).strip().title()
        if sev_title not in VALID_SEVERITIES:
            errors.append(f"Invalid severity '{severity}'. Allowed: {sorted(list(VALID_SEVERITIES))}")

        # Check for invalid numeric values
        for k, v in seed_conditions.items():
            if isinstance(v, (int, float)):
                if v < 0 and "growth" not in k.lower() and "change" not in k.lower():
                    errors.append(f"Suspicious negative value for seed condition '{k}': {v}")

        return len(errors) == 0, errors

    def validate_scenario(
        self,
        scenario_data: Any,
        expected_seeds: Dict[str, Any],
        expected_severity: str
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of generated compound scenario object.
        
        Returns:
            Dict containing:
                - 'is_valid': bool
                - 'status': 'passed' | 'flagged' | 'failed'
                - 'errors': List[str]
                - 'warnings': List[str]
                - 'metrics': Dict[str, Any]
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Check for empty or non-dict response
        if not scenario_data:
            errors.append("SLM response is empty or null.")
            return self._build_result(False, "failed", errors, warnings)

        if isinstance(scenario_data, str):
            try:
                scenario_data = json.loads(scenario_data)
            except Exception as exc:
                errors.append(f"Invalid structured JSON format: {exc}")
                return self._build_result(False, "failed", errors, warnings)

        if not isinstance(scenario_data, dict):
            errors.append(f"Expected dict schema, received: {type(scenario_data)}")
            return self._build_result(False, "failed", errors, warnings)

        # 2. Check required schema keys
        for field in REQUIRED_FIELDS:
            if field not in scenario_data or scenario_data[field] is None:
                errors.append(f"Missing required output field: '{field}'")

        # 3. Check severity alignment
        actual_severity = str(scenario_data.get("severity", "")).strip().title()
        exp_sev = str(expected_severity).strip().title()
        if actual_severity not in VALID_SEVERITIES:
            errors.append(f"Generated severity '{actual_severity}' is invalid.")
        elif actual_severity != exp_sev:
            errors.append(f"Severity mismatch: expected '{exp_sev}', received '{actual_severity}'.")

        patient_scenario = str(scenario_data.get("patient_scenario", "")).strip()
        if len(patient_scenario) < 30:
            errors.append(f"Scenario narrative is too brief or empty ({len(patient_scenario)} chars).")

        compound_interactions = scenario_data.get("compound_interactions", [])
        if not isinstance(compound_interactions, list) or len(compound_interactions) == 0:
            errors.append("Field 'compound_interactions' must be a non-empty list.")

        potential_risk_context = scenario_data.get("potential_risk_context", [])
        if not isinstance(potential_risk_context, list) or len(potential_risk_context) == 0:
            errors.append("Field 'potential_risk_context' must be a non-empty list.")

        # 4. Check Seed Condition Preservation & Omission
        combined_text = (
            patient_scenario + " " +
            " ".join(str(x) for x in compound_interactions) + " " +
            " ".join(str(x) for x in potential_risk_context)
        ).lower()

        preserved_count = 0
        total_seeds = len(expected_seeds)

        for seed_key, seed_val in expected_seeds.items():
            val_str = str(seed_val).strip()
            if not val_str or val_str.lower() in ("unknown", "none", "nan", "null"):
                total_seeds -= 1
                continue

            # Check if key tokens from seed value or seed field appear in the scenario
            clean_val = val_str.replace("_", " ")
            tokens = [t for t in re.split(r"[\s\-_/,\(\)]+", clean_val.lower()) if len(t) > 2 and not t.isdigit()]
            field_tokens = [t for t in re.split(r"[\s\-_/,\(\)]+", seed_key.lower()) if len(t) > 3]
            num_matches = re.findall(r"\b\d+(?:\.\d+)?\b", val_str)

            found = False
            # 1. Exact numeric value preservation (support both 100.0 and 100)
            if num_matches:
                for n in num_matches:
                    if re.search(r"\b" + re.escape(n) + r"\b", combined_text):
                        found = True
                        break
                    try:
                        f_val = float(n)
                        if f_val.is_integer():
                            int_str = str(int(f_val))
                            if re.search(r"\b" + re.escape(int_str) + r"\b", combined_text):
                                found = True
                                break
                        f_str = f"{f_val:.1f}"
                        if re.search(r"\b" + re.escape(f_str) + r"\b", combined_text):
                            found = True
                            break
                    except Exception:
                        pass

            # Roman numeral / stage equivalence
            if not found and "stage" in seed_key.lower():
                roman_map = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "i": "1", "ii": "2", "iii": "3", "iv": "4"}
                clean_stage = val_str.lower().replace("stage", "").strip()
                equiv = roman_map.get(clean_stage)
                if equiv and (re.search(r"\bstage\s+" + re.escape(equiv) + r"\b", combined_text) or re.search(r"\b" + re.escape(equiv) + r"\b", combined_text)):
                    found = True

            # 2. Categorical token preservation
            if not found and tokens:
                matched_tokens = sum(1 for t in tokens if t in combined_text)
                if matched_tokens >= 1 and (matched_tokens / len(tokens) >= 0.33 or len(tokens) == 1):
                    found = True

            # 3. Domain key presence (e.g. renal/creatinine, ctdna, tmb/mutation)
            if not found and field_tokens:
                if any(ft in combined_text for ft in field_tokens):
                    found = True

            if not found:
                if val_str.lower() in combined_text or clean_val.lower() in combined_text:
                    found = True

            if found:
                preserved_count += 1
            else:
                warnings.append(f"Seed condition '{seed_key}: {seed_val}' not prominently reflected in scenario.")

        preservation_rate = (preserved_count / total_seeds) if total_seeds > 0 else 1.0
        if preservation_rate < 0.35:
            errors.append(f"Scenario failed seed preservation threshold: only {preservation_rate:.1%} of seeds reflected.")

        # 5. Contradiction Detection
        # Check for opposite or contradictory statements
        for seed_key, seed_val in expected_seeds.items():
            k_lower = seed_key.lower()
            val_lower = str(seed_val).lower()
            if "renal" in k_lower or "creatinine" in k_lower:
                if any(imp in val_lower for imp in ["high", "elevat", "impair", "fail", "2.", "1.8", "1.9", "acute"]):
                    if "normal renal function" in combined_text or "unremarkable kidney" in combined_text:
                        errors.append("Contradiction: Seed indicates renal impairment, but scenario claims normal renal function.")
            if "ctdna" in k_lower:
                if any(r in val_lower for r in ["rising", "high", "elevat", "increase"]):
                    if "cleared ctdna" in combined_text or "undetectable ctdna" in combined_text:
                        errors.append("Contradiction: Seed indicates elevated/rising ctDNA, but scenario claims undetectable ctDNA.")

        # 6. Check for Unsupported Medical Claims / Fabricated Prescriptions
        prescriptive_patterns = [
            r"\bprescribe\s+(?:immediately|now|urgent)\b",
            r"\badminister\s+\d+\s*mg\s+of\b",
            r"\binitiate\s+experimental\s+drug\b"
        ]
        for pat in prescriptive_patterns:
            if re.search(pat, combined_text):
                warnings.append("Scenario contains prescriptive clinical directive (prohibited in decision-support prototype).")

        # 7. Duplicate Detection
        scenario_hash = hashlib.sha256(patient_scenario.encode("utf-8")).hexdigest()
        is_duplicate = scenario_hash in self.seen_hashes
        if is_duplicate:
            warnings.append("Duplicate scenario narrative detected (identical hash previously recorded).")
        else:
            self.seen_hashes.add(scenario_hash)

        # Status resolution
        is_valid = len(errors) == 0
        status = "passed" if is_valid and len(warnings) == 0 else ("flagged" if is_valid else "failed")

        metrics = {
            "preservation_rate": round(preservation_rate, 4),
            "total_seeds_checked": total_seeds,
            "seeds_preserved": preserved_count,
            "scenario_word_count": len(patient_scenario.split()),
            "compound_interactions_count": len(compound_interactions),
            "risk_factors_count": len(potential_risk_context),
            "is_duplicate": is_duplicate
        }

        return self._build_result(is_valid, status, errors, warnings, metrics)

    def _build_result(
        self,
        is_valid: bool,
        status: str,
        errors: List[str],
        warnings: List[str],
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "is_valid": is_valid,
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics or {}
        }
