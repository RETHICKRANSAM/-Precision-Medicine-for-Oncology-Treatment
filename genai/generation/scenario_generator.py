"""
Compound Scenario Generator (GenAI Stage).

Transforms structured oncology seed conditions into complete, realistic, coherent
compound patient scenarios using Qwen/Qwen2.5-0.5B-Instruct on CPU.
"""

import os
import sys
import time
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from genai.generation.prompt_builder import PromptBuilder
from genai.generation.scenario_validator import ScenarioValidator

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RESEARCH_DISCLAIMER = (
    "SYNTHETIC SCENARIO FOR RESEARCH AND DECISION-SUPPORT MODELING ONLY. "
    "DO NOT USE AS DIRECT CLINICAL ADVICE OR REAL-PATIENT MEDICAL PRESCRIPTION."
)


class CompoundScenarioGenerator:
    """SLM-based Compound Scenario Generator for Precision Oncology."""

    _instance = None
    _tokenizer = None
    _model = None

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.prompt_builder = PromptBuilder()
        self.validator = ScenarioValidator()
        self._load_model()

    def _load_model(self):
        """Loads Qwen2.5-0.5B-Instruct on CPU with lazy caching."""
        if CompoundScenarioGenerator._model is None:
            print(f"[CompoundScenarioGenerator] Loading model '{self.model_name}' on {self.device} (torch.float32)...")
            start = time.time()
            try:
                CompoundScenarioGenerator._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                CompoundScenarioGenerator._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                CompoundScenarioGenerator._model.eval()
                print(f"[CompoundScenarioGenerator] Model loaded successfully in {time.time() - start:.2f}s.")
            except Exception as exc:
                print(f"[CompoundScenarioGenerator] Warning: Could not load HuggingFace model ({exc}). Fallback generator enabled.")

        self.tokenizer = CompoundScenarioGenerator._tokenizer
        self.model = CompoundScenarioGenerator._model

    def _generate_text(self, messages: List[Dict[str, str]], max_new_tokens: int = 450) -> str:
        """Executes SLM inference on CPU with chat template."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer is not loaded.")

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def _extract_json_payload(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from raw SLM response."""
        # 1. Look for ```json ... ``` markdown fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # 2. Look for first { to last }
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_substr = raw_text[start:end + 1]
            try:
                return json.loads(json_substr)
            except Exception:
                pass

        return None

    def _synthesize_deterministic_scenario(
        self,
        seed_conditions: Dict[str, Any],
        severity: str,
        scenario_id: str
    ) -> Dict[str, Any]:
        """
        Deterministic, faithful clinical scenario synthesis engine.
        Ensures 100% preservation of all seed variables and valid schema structure.
        """
        sev = severity.title()
        seeds = seed_conditions

        # Primary clinical variables
        tmb = seeds.get("Tumor Mutation Burden") or seeds.get("Tumor Marker Level") or "baseline"
        ctdna = seeds.get("ctDNA Level") or seeds.get("ctDNA Trend") or seeds.get("ctDNA Level/Trend") or "monitored"
        renal = seeds.get("Renal Function (Serum Creatinine)") or seeds.get("Renal Function") or "stable"
        organ = seeds.get("Organ Site Involvement") or seeds.get("Organ Involvement") or seeds.get("Cancer Type") or "solid tumor"
        mutation = seeds.get("Genomic Mutation") or "unspecified alteration"
        drug = seeds.get("Current Antineoplastic Drug") or seeds.get("Antineoplastic Drug") or "targeted therapy"
        dose = seeds.get("Drug Dosage (mg)") or seeds.get("Dosage Level") or "standard dose"
        ae = seeds.get("Reported Adverse Events") or "None Reported"
        symptoms = seeds.get("Reported Symptoms") or "mild fatigue"
        stage = seeds.get("Cancer Stage") or "advanced"

        # Extended clinical panels
        age = seeds.get("Patient Age")
        sex = seeds.get("Patient Sex")
        group = seeds.get("Patient Group")
        comorb = seeds.get("Organ/System Comorbidities")
        adherence = seeds.get("Treatment Adherence Pct")
        prior_tx = seeds.get("Prior Therapies")
        response = seeds.get("Treatment Response")
        tox_score = seeds.get("Toxicity Score")
        tox_risk = seeds.get("Baseline Toxicity Risk")
        risk_score = seeds.get("Risk Score")
        symptom_count = seeds.get("Symptom Count")
        bp = seeds.get("Systolic Bp")
        hr = seeds.get("Heart Rate")
        o2 = seeds.get("Oxygen Saturation")
        temp = seeds.get("Temperature")
        bili = seeds.get("Total Bilirubin (Hepatic)")
        alt = seeds.get("ALT (Hepatic Enzyme)")
        ast = seeds.get("AST (Hepatic Enzyme)")
        wbc = seeds.get("WBC Count")
        hgb = seeds.get("Hemoglobin Level")
        plt = seeds.get("Platelet Count")
        egfr = seeds.get("Egfr Expression")
        kras = seeds.get("Kras Expression")
        alk = seeds.get("Alk Expression")
        urgency = seeds.get("Triage Urgency Status")

        # Build panel narrative clauses if present
        extended_clauses = []
        if age is not None or sex is not None or group is not None or comorb is not None:
            demo_parts = []
            if age is not None and sex is not None:
                demo_parts.append(f"{age}-year-old {sex}")
            if group is not None:
                demo_parts.append(f"Patient Group: {group}")
            if comorb is not None:
                demo_parts.append(f"comorbidities: {comorb}")
            if demo_parts:
                extended_clauses.append(f"Patient profile: {', '.join(demo_parts)}.")

        if bp is not None or hr is not None or o2 is not None or temp is not None:
            v_parts = []
            if bp is not None: v_parts.append(f"systolic blood pressure {bp} mmHg")
            if hr is not None: v_parts.append(f"heart rate {hr} bpm")
            if o2 is not None: v_parts.append(f"oxygen saturation {o2}%")
            if temp is not None: v_parts.append(f"temperature {temp} F")
            if v_parts:
                extended_clauses.append(f"Vital signs record {', '.join(v_parts)}.")

        if egfr is not None or kras is not None or alk is not None:
            mol_parts = []
            if egfr is not None: mol_parts.append(f"EGFR expression {egfr}")
            if kras is not None: mol_parts.append(f"KRAS expression {kras}")
            if alk is not None: mol_parts.append(f"ALK expression {alk}")
            if mol_parts:
                extended_clauses.append(f"Molecular panel indicates {', '.join(mol_parts)}.")

        if bili is not None or alt is not None or ast is not None or wbc is not None or hgb is not None or plt is not None:
            lab_parts = []
            if bili is not None: lab_parts.append(f"total bilirubin {bili} mg/dL")
            if alt is not None: lab_parts.append(f"ALT {alt} U/L")
            if ast is not None: lab_parts.append(f"AST {ast} U/L")
            if wbc is not None: lab_parts.append(f"WBC count {wbc} K/uL")
            if hgb is not None: lab_parts.append(f"hemoglobin {hgb} g/dL")
            if plt is not None: lab_parts.append(f"platelet count {plt} K/uL")
            if lab_parts:
                extended_clauses.append(f"Laboratory assessment registers {', '.join(lab_parts)}.")

        if adherence is not None or prior_tx is not None or response is not None:
            tx_parts = []
            if prior_tx is not None: tx_parts.append(f"{prior_tx} prior lines of therapy")
            if adherence is not None: tx_parts.append(f"{adherence}% treatment adherence")
            if response is not None: tx_parts.append(f"treatment response: {response}")
            if tx_parts:
                extended_clauses.append(f"Treatment status: {', '.join(tx_parts)}.")

        if tox_risk is not None or tox_score is not None or risk_score is not None or symptom_count is not None or urgency is not None:
            risk_parts = []
            if tox_risk is not None: risk_parts.append(f"baseline toxicity risk: {tox_risk}")
            if tox_score is not None: risk_parts.append(f"toxicity score: {tox_score}")
            if risk_score is not None: risk_parts.append(f"risk score: {risk_score}")
            if symptom_count is not None: risk_parts.append(f"symptom count: {symptom_count}")
            if urgency is not None: risk_parts.append(f"triage urgency: {urgency}")
            if risk_parts:
                extended_clauses.append(f"Risk metrics: {', '.join(risk_parts)}.")

        # Collect any uncaptured seeds to guarantee 100% preservation
        covered_keys = {
            "Tumor Mutation Burden", "Tumor Marker Level", "ctDNA Level", "ctDNA Trend", "ctDNA Level/Trend",
            "Renal Function (Serum Creatinine)", "Renal Function", "Organ Site Involvement", "Organ Involvement",
            "Cancer Type", "Genomic Mutation", "Current Antineoplastic Drug", "Antineoplastic Drug",
            "Drug Dosage (mg)", "Dosage Level", "Reported Adverse Events", "Reported Symptoms", "Cancer Stage",
            "Patient Age", "Patient Sex", "Patient Group", "Organ/System Comorbidities", "Treatment Adherence Pct",
            "Prior Therapies", "Treatment Response", "Toxicity Score", "Baseline Toxicity Risk", "Risk Score",
            "Symptom Count", "Systolic Bp", "Heart Rate", "Oxygen Saturation", "Temperature",
            "Total Bilirubin (Hepatic)", "ALT (Hepatic Enzyme)", "AST (Hepatic Enzyme)", "WBC Count",
            "Hemoglobin Level", "Platelet Count", "Egfr Expression", "Kras Expression", "Alk Expression",
            "Triage Urgency Status", "Clinical Report"
        }
        uncaptured = []
        for k, v in seeds.items():
            if k not in covered_keys and str(v).strip().lower() not in ("unknown", "none", "nan", "null", ""):
                uncaptured.append(f"{k}: {v}")
        if uncaptured:
            extended_clauses.append(f"Additional clinical parameters: {', '.join(uncaptured)}.")

        extended_text = " " + " ".join(extended_clauses) if extended_clauses else ""
        stage_str = f"Stage {stage}" if not str(stage).lower().startswith("stage") else str(stage)

        if sev == "Mild":
            scenario_narrative = (
                f"The patient presents with {stage_str} {organ} harboring {mutation}, currently undergoing evaluation while on {drug} ({dose}). "
                f"Biomarker kinetics reveal a controlled ctDNA status ({ctdna}) accompanied by tumor marker activity of {tmb}. "
                f"Renal clearance remains preserved with serum creatinine at {renal}, allowing adequate drug tolerability despite {symptoms} and reported adverse events ({ae})."
                f"{extended_text}"
            )
            interactions = [
                f"Controlled ctDNA dynamic ({ctdna}) correlates with manageable tumor burden ({tmb}), indicating early disease stabilization in {organ}.",
                f"Preserved renal filtration ({renal}) facilitates normal pharmacokinetic clearance of {drug} without dose-limiting nephrotoxicity."
            ]
            risks = [
                f"Low acute systemic toxicity risk; continue routine longitudinal ctDNA monitoring.",
                f"Mild symptomatology ({symptoms}) and adverse manifestations ({ae}) require standard supportive care and hydration."
            ]

        elif sev == "Moderate":
            scenario_narrative = (
                f"The patient with {stage_str} {organ} harboring {mutation} is undergoing ongoing systemic therapy with {drug} ({dose}). "
                f"Recent clinical surveillance demonstrates evolving disease kinetics with ctDNA at {ctdna} and tumor burden index at {tmb}. "
                f"Renal function parameters indicate borderline clearance ({renal}), interacting with reported {symptoms} and adverse manifestations of {ae}."
                f"{extended_text}"
            )
            interactions = [
                f"Active tumor burden ({tmb}) coupled with ctDNA level ({ctdna}) suggests persistent cellular turnover in {organ}.",
                f"Renal clearance level ({renal}) combined with {drug} administration compounds the risk of escalating {ae}."
            ]
            risks = [
                f"Potential for progressive organ compromise if renal parameter ({renal}) deteriorates further during treatment.",
                f"Surveillance required for adverse event exacerbation ({ae}) under current {dose} dosing schedule."
            ]

        elif sev == "Severe":
            scenario_narrative = (
                f"The patient presents in a high-urgency oncology status with {stage_str} {organ} ({mutation}) undergoing therapy with {drug} ({dose}). "
                f"Markedly elevated ctDNA dynamics ({ctdna}) and high tumor mutation/marker burden ({tmb}) reflect aggressive disease proliferation. "
                f"Concurrent renal stress with elevated creatinine ({renal}) severely impairs metabolic clearance, compounding acute {symptoms} and severe {ae}."
                f"{extended_text}"
            )
            interactions = [
                f"Critical synergistic stress: aggressive biomarker kinetics ({ctdna}, {tmb}) coincide with compromised renal reserve ({renal}).",
                f"Impaired clearance ({renal}) accelerates systemic accumulation of {drug}, directly precipitating high-grade {ae}."
            ]
            risks = [
                f"High risk of multi-organ decompensation and treatment-related toxic shock requiring immediate clinical team review.",
                f"Urgent indication for therapeutic drug monitoring and supportive renoprotective countermeasures."
            ]

        else:  # Wildcard
            scenario_narrative = (
                f"The patient exhibits an atypical clinical course in {stage_str} {organ} ({mutation}) receiving {drug} ({dose}). "
                f"Surveillance reveals a discordant biomarker profile characterized by unusual ctDNA kinetics ({ctdna}) despite tumor burden of {tmb}. "
                f"Renal parameters register at {renal}, displaying an idiosyncratic physiological response characterized by sudden {symptoms} and atypical {ae}."
                f"{extended_text}"
            )
            interactions = [
                f"Atypical biomarker divergence: discordant ctDNA pattern ({ctdna}) relative to tumor marker level ({tmb}) indicates clonal heterogeneity.",
                f"Idiosyncratic interaction between {drug} metabolism, renal function ({renal}), and emergent {ae}."
            ]
            risks = [
                f"Unconventional clinical trajectory requiring specialized molecular tumor board review and repeat genomic reassessment.",
                f"Risk of misinterpreting pseudo-progression versus genuine clonal escape under {drug}."
            ]

        return {
            "scenario_id": scenario_id,
            "seed_conditions": seed_conditions,
            "severity": sev,
            "patient_scenario": scenario_narrative,
            "compound_interactions": interactions,
            "potential_risk_context": risks,
            "generation_metadata": {
                "model": self.model_name,
                "prompt_version": self.prompt_builder.prompt_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "synthesized_deterministic",
                "disclaimer": RESEARCH_DISCLAIMER
            },
            "validation": {
                "status": "pending_validation",
                "errors": []
            }
        }

    def _ground_and_preserve_seeds(
        self,
        payload: Dict[str, Any],
        expected_seeds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inspects scenario payload and explicitly preserves any omitted seed conditions
        to guarantee comprehensive preservation without modifying clinical meaning.
        """
        if not payload or not isinstance(payload, dict):
            return payload

        combined_text = (
            str(payload.get("patient_scenario", "")) + " " +
            " ".join(str(x) for x in payload.get("compound_interactions", [])) + " " +
            " ".join(str(x) for x in payload.get("potential_risk_context", []))
        ).lower()

        missing_items = []
        for seed_key, seed_val in expected_seeds.items():
            val_str = str(seed_val).strip()
            if not val_str or val_str.lower() in ("unknown", "none", "nan", "null"):
                continue

            clean_val = val_str.replace("_", " ")
            tokens = [t for t in re.split(r"[\s\-_/,\(\)]+", clean_val.lower()) if len(t) > 2 and not t.isdigit()]
            field_tokens = [t for t in re.split(r"[\s\-_/,\(\)]+", seed_key.lower()) if len(t) > 3]
            num_matches = re.findall(r"\b\d+(?:\.\d+)?\b", val_str)

            found = False
            if num_matches:
                for n in num_matches:
                    if re.search(r"\b" + re.escape(n) + r"\b", combined_text):
                        found = True
                        break
                    try:
                        f_val = float(n)
                        if f_val.is_integer() and re.search(r"\b" + re.escape(str(int(f_val))) + r"\b", combined_text):
                            found = True
                            break
                        f_str = f"{f_val:.1f}"
                        if re.search(r"\b" + re.escape(f_str) + r"\b", combined_text):
                            found = True
                            break
                    except Exception:
                        pass

            if not found and tokens:
                matched = sum(1 for t in tokens if t in combined_text)
                if matched >= 1 and (matched / len(tokens) >= 0.33 or len(tokens) == 1):
                    found = True

            if not found and field_tokens:
                if any(ft in combined_text for ft in field_tokens):
                    found = True

            if not found:
                if val_str.lower() in combined_text or clean_val.lower() in combined_text:
                    found = True

            if not found:
                missing_items.append((seed_key, seed_val))

        if not missing_items:
            return payload

        med_clauses = []
        triage_clauses = []
        genomic_clauses = []
        lab_clauses = []
        other_clauses = []

        for k, v in missing_items:
            k_lower = k.lower()
            v_str = str(v).strip()
            if "drug" in k_lower or "dosage" in k_lower or "dose" in k_lower:
                med_clauses.append(f"{k}: {v_str}")
            elif "urgency" in k_lower or "triage" in k_lower:
                triage_clauses.append(f"triage urgency evaluated as {v_str}")
            elif "mutation" in k_lower or "expression" in k_lower:
                genomic_clauses.append(f"{k} {v_str}")
            elif any(term in k_lower for term in ["bilirubin", "alt", "ast", "wbc", "hemoglobin", "platelet", "bp", "heart", "oxygen", "temp", "creatinine"]):
                lab_clauses.append(f"{k} {v_str}")
            else:
                other_clauses.append(f"{k}: {v_str}")

        grounding_clauses = []
        if med_clauses:
            grounding_clauses.append(f"Antineoplastic therapy specifications record {', '.join(med_clauses)}.")
        if triage_clauses:
            grounding_clauses.append(f"Clinical intake records {', '.join(triage_clauses)}.")
        if genomic_clauses:
            grounding_clauses.append(f"Molecular profiling confirms {', '.join(genomic_clauses)}.")
        if lab_clauses:
            grounding_clauses.append(f"Baseline clinical laboratory measurements register: {', '.join(lab_clauses)}.")
        if other_clauses:
            grounding_clauses.append(f"Documented patient parameters include: {', '.join(other_clauses)}.")

        supplement_text = " " + " ".join(grounding_clauses)
        narrative = str(payload.get("patient_scenario", "")).rstrip()
        payload["patient_scenario"] = narrative + supplement_text

        return payload

    def generate_scenario(
        self,
        seed_conditions: Union[Dict[str, Any], pd.Series],
        severity: str = "Moderate",
        scenario_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a validated compound patient scenario from structured seed conditions.
        
        Args:
            seed_conditions: Dictionary or Series of structured clinical/genomic seed fields.
            severity: One of 'Mild', 'Moderate', 'Severe', 'Wildcard'.
            scenario_id: Optional unique scenario identifier.
            
        Returns:
            Dict conforming to the required structured output schema.
        """
        # 1. Clean & extract seed conditions
        try:
            cleaned_seeds = self.prompt_builder.extract_seed_conditions(seed_conditions)
        except Exception as exc:
            # Missing or completely invalid seed record
            return {
                "scenario_id": scenario_id or "SCEN-ERROR",
                "seed_conditions": {},
                "severity": severity,
                "patient_scenario": "",
                "compound_interactions": [],
                "potential_risk_context": [],
                "generation_metadata": {
                    "model": self.model_name,
                    "prompt_version": self.prompt_builder.prompt_version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "seed_extraction_error"
                },
                "validation": {
                    "status": "failed",
                    "is_valid": False,
                    "errors": [f"Seed extraction error: {exc}"]
                }
            }

        # 2. Pre-validate seeds and severity
        is_valid_seed, seed_errors = self.validator.validate_seed_input(cleaned_seeds, severity)
        if not is_valid_seed:
            return {
                "scenario_id": scenario_id or "SCEN-INVALID",
                "seed_conditions": cleaned_seeds,
                "severity": severity,
                "patient_scenario": "",
                "compound_interactions": [],
                "potential_risk_context": [],
                "generation_metadata": {
                    "model": self.model_name,
                    "prompt_version": self.prompt_builder.prompt_version,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "invalid_input"
                },
                "validation": {
                    "status": "failed",
                    "is_valid": False,
                    "errors": seed_errors
                }
            }

        # 3. Build Prompt
        prompt_data = self.prompt_builder.build_prompt(cleaned_seeds, severity, scenario_id)
        scen_id = prompt_data["scenario_id"]

        # 4. SLM Inference
        raw_response = None
        parsed_payload = None
        generation_status = "model_generated"

        if self.model is not None and self.tokenizer is not None:
            try:
                raw_response = self._generate_text(prompt_data["messages"], max_new_tokens=450)
                parsed_payload = self._extract_json_payload(raw_response)
            except Exception as exc:
                print(f"[CompoundScenarioGenerator] Model inference error ({exc}). Using deterministic synthesis.")

        # Fallback to high-fidelity deterministic synthesis if SLM output was non-JSON or unavailable
        if not parsed_payload or not isinstance(parsed_payload, dict):
            parsed_payload = self._synthesize_deterministic_scenario(cleaned_seeds, severity, scen_id)
            generation_status = "deterministic_synthesized"
        else:
            # Ensure required top-level structure and seeds are properly attached
            parsed_payload["scenario_id"] = scen_id
            parsed_payload["seed_conditions"] = cleaned_seeds
            parsed_payload["severity"] = prompt_data["severity"]
            parsed_payload["generation_metadata"] = {
                "model": self.model_name,
                "prompt_version": self.prompt_builder.prompt_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": generation_status,
                "disclaimer": RESEARCH_DISCLAIMER
            }

        # Ensure compound_interactions and potential_risk_context have structured points
        interactions = parsed_payload.get("compound_interactions", [])
        if isinstance(interactions, list) and len(interactions) == 1:
            single = str(interactions[0])
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", single) if len(s.strip()) > 15]
            if len(sents) >= 2:
                parsed_payload["compound_interactions"] = sents

        risks = parsed_payload.get("potential_risk_context", [])
        if isinstance(risks, list) and len(risks) == 1:
            single = str(risks[0])
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", single) if len(s.strip()) > 15]
            if len(sents) >= 2:
                parsed_payload["potential_risk_context"] = sents

        # 5. Multi-Tier Validation & Seed Grounding
        parsed_payload = self._ground_and_preserve_seeds(parsed_payload, cleaned_seeds)
        val_result = self.validator.validate_scenario(parsed_payload, cleaned_seeds, prompt_data["severity"])

        # If model generation failed validation checks, use deterministic fallback to guarantee quality
        if not val_result["is_valid"]:
            parsed_payload = self._synthesize_deterministic_scenario(cleaned_seeds, severity, scen_id)
            parsed_payload = self._ground_and_preserve_seeds(parsed_payload, cleaned_seeds)
            val_result = self.validator.validate_scenario(parsed_payload, cleaned_seeds, prompt_data["severity"])

        parsed_payload["validation"] = {
            "status": val_result["status"],
            "is_valid": val_result["is_valid"],
            "errors": val_result["errors"],
            "warnings": val_result["warnings"],
            "metrics": val_result.get("metrics", {})
        }

        return parsed_payload

    def generate_batch(
        self,
        records: Union[pd.DataFrame, List[Dict[str, Any]]],
        severities: Optional[List[str]] = None,
        output_jsonl: Optional[str] = None,
        output_csv: Optional[str] = None,
        log_csv: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes batch scenario generation across multiple seed records.
        Saves output to JSONL and CSV if paths are provided.
        """
        if isinstance(records, pd.DataFrame):
            data_list = records.to_dict(orient="records")
        else:
            data_list = list(records)

        severity_rotation = ["Mild", "Moderate", "Severe", "Wildcard"]
        results = []
        logs = []

        print(f"[CompoundScenarioGenerator] Starting batch generation for {len(data_list)} seed records...")
        start_time = time.time()

        for idx, row in enumerate(data_list):
            item_start = time.time()
            # Determine severity
            if severities and idx < len(severities):
                sev = severities[idx]
            else:
                sev = severity_rotation[idx % len(severity_rotation)]

            scenario_id = f"SCEN-BATCH-{idx+1:04d}"
            scenario = self.generate_scenario(row, severity=sev, scenario_id=scenario_id)
            elapsed = time.time() - item_start

            results.append(scenario)

            # Record audit log row
            val_info = scenario.get("validation", {})
            logs.append({
                "scenario_id": scenario_id,
                "severity": sev,
                "status": val_info.get("status", "unknown"),
                "is_valid": val_info.get("is_valid", False),
                "preservation_rate": val_info.get("metrics", {}).get("preservation_rate", 0.0),
                "word_count": val_info.get("metrics", {}).get("scenario_word_count", 0),
                "latency_seconds": round(elapsed, 3),
                "error_count": len(val_info.get("errors", [])),
                "warning_count": len(val_info.get("warnings", []))
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(data_list):
                print(f"  Processed {idx + 1}/{len(data_list)} scenarios ({elapsed:.2f}s/item)...")

        total_elapsed = time.time() - start_time
        print(f"[CompoundScenarioGenerator] Batch completed in {total_elapsed:.2f}s (Avg: {total_elapsed/len(data_list):.2f}s/scen).")

        # Save JSONL if requested
        if output_jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"[CompoundScenarioGenerator] Saved {len(results)} scenarios to {output_jsonl}")

        # Save CSV if requested
        if output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            flat_rows = []
            for r in results:
                flat_rows.append({
                    "scenario_id": r.get("scenario_id"),
                    "severity": r.get("severity"),
                    "patient_scenario": r.get("patient_scenario"),
                    "compound_interactions": " | ".join(r.get("compound_interactions", [])),
                    "potential_risk_context": " | ".join(r.get("potential_risk_context", [])),
                    "validation_status": r.get("validation", {}).get("status"),
                    "is_valid": r.get("validation", {}).get("is_valid"),
                    "seed_conditions_json": json.dumps(r.get("seed_conditions", {}), ensure_ascii=False),
                    "timestamp": r.get("generation_metadata", {}).get("timestamp")
                })
            pd.DataFrame(flat_rows).to_csv(output_csv, index=False, encoding="utf-8")
            print(f"[CompoundScenarioGenerator] Saved scenarios CSV to {output_csv}")

        # Save Log CSV if requested
        if log_csv:
            os.makedirs(os.path.dirname(os.path.abspath(log_csv)), exist_ok=True)
            pd.DataFrame(logs).to_csv(log_csv, index=False, encoding="utf-8")
            print(f"[CompoundScenarioGenerator] Saved execution logs to {log_csv}")

        return results
