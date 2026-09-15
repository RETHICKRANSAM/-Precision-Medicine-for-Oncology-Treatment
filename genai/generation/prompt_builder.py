"""
Prompt Builder for the GenAI SLM-Based Compound Scenario Generator.

Builds parameterized, versioned prompts that dynamically insert all available
structured oncology seed conditions without hard-coding or value modification.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

DEFAULT_PROMPT_VERSION = "v1.0"
ALLOWED_SEVERITIES = {"mild", "moderate", "severe", "wildcard"}

# Canonical mapping for common variations of seed fields (supports both underscores and spaces)
CANONICAL_FIELD_NAMES = {
    "tmb": "Tumor Mutation Burden",
    "tumor_mutation_burden": "Tumor Mutation Burden",
    "tumor mutation burden": "Tumor Mutation Burden",
    "tumor_marker": "Tumor Marker Level",
    "tumor marker": "Tumor Marker Level",
    "tumor_marker_level": "Tumor Marker Level",
    "ctdna": "ctDNA Level/Trend",
    "ctdna_level": "ctDNA Level",
    "ctdna level": "ctDNA Level",
    "ctdna_trend": "ctDNA Trend",
    "ctdna trend": "ctDNA Trend",
    "creatinine": "Renal Function (Serum Creatinine)",
    "renal_function": "Renal Function",
    "renal function": "Renal Function",
    "organ_site": "Organ Site Involvement",
    "organ site": "Organ Site Involvement",
    "organ_involvement": "Organ Involvement",
    "organ involvement": "Organ Involvement",
    "comorbidities": "Organ/System Comorbidities",
    "cancer_type": "Cancer Type",
    "cancer type": "Cancer Type",
    "cancer_stage": "Cancer Stage",
    "cancer stage": "Cancer Stage",
    "gene_mutation": "Genomic Mutation",
    "gene mutation": "Genomic Mutation",
    "genomic mutation": "Genomic Mutation",
    "treatment_drug": "Current Antineoplastic Drug",
    "treatment drug": "Current Antineoplastic Drug",
    "drug_name": "Antineoplastic Drug",
    "drug name": "Antineoplastic Drug",
    "dosage_mg": "Drug Dosage (mg)",
    "dosage mg": "Drug Dosage (mg)",
    "dosage_level": "Dosage Level",
    "dosage level": "Dosage Level",
    "toxicity_risk": "Baseline Toxicity Risk",
    "toxicity risk": "Baseline Toxicity Risk",
    "toxicity_score": "Toxicity Score",
    "risk_score": "Risk Score",
    "symptom_report": "Reported Symptoms",
    "symptom report": "Reported Symptoms",
    "symptom_text": "Reported Symptoms",
    "symptom text": "Reported Symptoms",
    "adverse_event": "Reported Adverse Events",
    "adverse event": "Reported Adverse Events",
    "urgency": "Triage Urgency Status",
    "age": "Patient Age",
    "sex": "Patient Sex",
    "wbc_count": "WBC Count",
    "hemoglobin": "Hemoglobin Level",
    "platelet_count": "Platelet Count",
    "bilirubin": "Total Bilirubin (Hepatic)",
    "total_bilirubin": "Total Bilirubin (Hepatic)",
    "total bilirubin": "Total Bilirubin (Hepatic)",
    "alt": "ALT (Hepatic Enzyme)",
    "ast": "AST (Hepatic Enzyme)",
    "clinical_report": "Clinical Report",
    "clinical report": "Clinical Report"
}


class PromptBuilder:
    """Dynamically builds parameterized prompts for compound scenario generation."""

    def __init__(self, prompt_dir: Optional[str] = None):
        if prompt_dir is None:
            prompt_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
        self.prompt_dir = os.path.abspath(prompt_dir)
        self.prompt_version = self._load_prompt_version()
        self.system_prompt, self.user_template = self._load_prompt_templates()

    def _load_prompt_version(self) -> str:
        version_file = os.path.join(self.prompt_dir, "prompt_version.txt")
        if os.path.exists(version_file):
            try:
                with open(version_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("Prompt Version:"):
                            return line.split(":", 1)[1].strip()
            except Exception:
                pass
        return DEFAULT_PROMPT_VERSION

    def _load_prompt_templates(self) -> Tuple[str, str]:
        prompt_file = os.path.join(self.prompt_dir, "compound_scenario_prompt.txt")
        default_system = (
            "You are an expert Clinical Oncology Systems Physiologist and Scenario Modeling Specialist. "
            "Generate a realistic, coherent, and clinically grounded COMPOUND PATIENT SCENARIO from structured seed conditions. "
            "Explain how the conditions interact physiologically. Preserve all original seed values exactly. "
            "Output strictly valid JSON matching the schema."
        )
        default_user = (
            "Generate a compound oncology patient scenario using the following structured seed conditions.\n\n"
            "Target Severity Profile: {severity}\n\n"
            "Structured Seed Conditions:\n{formatted_seed_conditions}\n\n"
            "Output strictly valid JSON matching the schema."
        )

        if not os.path.exists(prompt_file):
            return default_system, default_user

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

            sys_part = ""
            user_part = ""
            if "[SYSTEM_PROMPT]" in content:
                parts = content.split("[SYSTEM_PROMPT]", 1)[1]
                if "[OUTPUT_SCHEMA]" in parts:
                    sys_part, rest = parts.split("[OUTPUT_SCHEMA]", 1)
                    if "[USER_PROMPT_TEMPLATE]" in rest:
                        schema_part, user_part = rest.split("[USER_PROMPT_TEMPLATE]", 1)
                        sys_part = sys_part.strip() + "\n\n[REQUIRED JSON SCHEMA]:\n" + schema_part.strip()
                        user_part = user_part.strip()
            if sys_part and user_part:
                return sys_part, user_part
        except Exception:
            pass

        return default_system, default_user

    def extract_seed_conditions(self, raw_record: Any) -> Dict[str, Any]:
        """
        Extracts all non-null, valid seed conditions from dict or pandas Series.
        Preserves original values without invention or modification.
        """
        if hasattr(raw_record, "to_dict"):
            data = raw_record.to_dict()
        elif isinstance(raw_record, dict):
            data = dict(raw_record)
        else:
            raise TypeError(f"Unsupported record type: {type(raw_record)}. Expected dict or Series.")

        # Filter out internal keys, targets, or redundant free-text notes
        excluded_keys = {
            "patient_id", "encounter_id", "encounter_date", "clinical_note",
            "clinical_notes", "summary", "ner", "image_path", "image_id",
            "ct_scan_path", "mri_scan_path", "histopathology_image_path",
            "image_type", "label", "dataset_split"
        }

        extracted: Dict[str, Any] = {}
        for k, v in data.items():
            k_clean = str(k).strip()
            k_lower = k_clean.lower()
            if k_lower in excluded_keys:
                continue

            # Ignore NaN, None, or empty strings
            if v is None:
                continue
            if isinstance(v, float) and (v != v):  # NaN check
                continue
            v_str = str(v).strip()
            if v_str == "" or v_str.lower() in ("nan", "none", "null"):
                continue

            # Store canonical display name
            if k_lower in CANONICAL_FIELD_NAMES:
                canonical_name = CANONICAL_FIELD_NAMES[k_lower]
            elif any(c.isupper() for c in k_clean) and " " in k_clean:
                canonical_name = k_clean
            else:
                canonical_name = k_clean.replace("_", " ").title()

            extracted[canonical_name] = v

        if not extracted:
            raise ValueError("Missing seed conditions: record is empty or contains no valid structured fields.")

        return extracted

    def format_seed_conditions(self, seed_conditions: Dict[str, Any]) -> str:
        """Formats seed conditions dictionary into structured text lines for the prompt."""
        lines = []
        for field, value in seed_conditions.items():
            lines.append(f"- {field}: {value}")
        return "\n".join(lines)

    def build_prompt(
        self,
        seed_conditions: Dict[str, Any],
        severity: str = "Moderate",
        scenario_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds the complete parameterized prompt payload.
        
        Returns:
            Dict with keys:
                - 'system_prompt': str
                - 'user_prompt': str
                - 'messages': List[Dict[str, str]]
                - 'severity': str
                - 'scenario_id': str
                - 'formatted_seeds': str
                - 'prompt_version': str
        """
        sev_clean = str(severity).strip().title()
        if sev_clean.lower() not in ALLOWED_SEVERITIES:
            raise ValueError(f"Invalid severity '{severity}'. Must be one of: Mild, Moderate, Severe, Wildcard.")

        if not scenario_id:
            scenario_id = f"SCEN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:17]}"

        formatted_seeds = self.format_seed_conditions(seed_conditions)
        seeds_json_sample = json.dumps(seed_conditions, indent=2, ensure_ascii=False)

        # Substitute user prompt template
        user_prompt = (
            f"Generate a compound oncology patient scenario using the following structured seed conditions.\n\n"
            f"Target Severity Profile: {sev_clean}\n"
            f"Scenario ID: {scenario_id}\n\n"
            f"Structured Seed Conditions:\n"
            f"{formatted_seeds}\n\n"
            f"Clinical Directives:\n"
            f"1. Explain physiological interactions between the conditions.\n"
            f"2. Explicitly preserve and mention every supplied seed condition (including exact drug, dosage, genomic mutation, triage status, and laboratory/vital parameters) in the scenario narrative, compound interactions, or potential risk context.\n"
            f"3. Preserve every seed value exactly (no modifications, alterations, or inventions).\n"
            f"4. Adhere to the {sev_clean} severity profile.\n"
            f"5. Output STRICTLY a valid JSON object matching the required schema."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "messages": messages,
            "severity": sev_clean,
            "scenario_id": scenario_id,
            "formatted_seeds": formatted_seeds,
            "prompt_version": self.prompt_version,
            "seed_conditions": seed_conditions
        }
