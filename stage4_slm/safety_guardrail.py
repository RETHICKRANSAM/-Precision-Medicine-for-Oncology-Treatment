"""
Safety and Hallucination Guardrail Layer for Stage 04 SLM Clinical Summarization.

Verifies factual consistency between the source clinical report and the generated summary.
Detects potential hallucinations or unsupported entities:
- Prescribed medications / antineoplastic drugs
- Dosages and regimens
- Genomic mutations / biomarkers
- Adverse events and toxicities
- Clinical symptoms
- Unsupported diagnostic / treatment recommendations
"""

import re
from typing import Dict, Any, List, Optional

KNOWN_DRUGS = [
    "osimertinib", "erlotinib", "gefitinib", "afatinib", "dacomitinib",
    "sotorasib", "adagrasib",
    "alectinib", "crizotinib", "ceritinib", "brigatinib", "lorlatinib",
    "dabrafenib", "trametinib", "vemurafenib",
    "cisplatin", "carboplatin", "pemetrexed", "paclitaxel", "docetaxel", "gemcitabine",
    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab"
]

KNOWN_MUTATIONS = [
    "egfr l858r", "egfr exon 19 del", "egfr t790m", "egfr exon 20 ins",
    "kras g12c", "kras g12d", "kras g12v", "kras",
    "alk fusion", "alk rearrangement", "alk",
    "ros1 fusion", "ros1",
    "braf v600e", "braf",
    "met amplification", "met exon 14 skipping", "met",
    "ret fusion", "ret",
    "her2 mutation", "her2 amplification", "her2",
    "stk11 mutation", "stk11 mut", "stk11",
    "tp53 mutation", "tp53",
    "keap1 mutation", "keap1"
]

KNOWN_ADVERSE_EVENTS = [
    "hepatotoxicity", "mucositis", "thrombocytopenia", "neutropenia",
    "peripheral neuropathy", "pneumonitis", "severe diarrhea", "diarrhea",
    "colitis", "nephrotoxicity", "rash", "fatigue", "fever", "dyspnea",
    "nausea", "vomiting", "anemia", "chills"
]

KNOWN_SYMPTOMS = [
    "shortness of breath", "chest discomfort", "persistent cough", "cough",
    "loss of appetite", "mild fatigue", "skin rash", "rash worsening",
    "nausea after treatment", "fever and chills", "headache", "dizziness",
    "difficulty swallowing", "hemoptysis", "bone pain", "weight loss"
]

RECOMMENDATION_TRIGGERS = [
    "should start", "recommend starting", "should discontinue", "recommend stopping",
    "switch to", "increase dose", "decrease dose", "titrate to",
    "initiate therapy", "prescribe", "administer immediately", "advised to take",
    "patient should be treated with", "treatment recommendation", "plan of care requires",
    "indicated treatment is", "urgent prescription", "start patient on"
]

UNAUTHORIZED_DIAGNOSES = [
    "confirms definitive diagnosis", "diagnosed as terminal", "autonomous diagnosis",
    "confirmed stage iv progression", "clinical verdict confirms", "pathologist concludes definitively",
    "newly diagnosed with metastatic"
]


class ClinicalSummarySafetyGuardrail:
    """Post-generation clinical validation and safety guardrail."""
    
    DISCLAIMER = "Generated summary — verify against the source clinical report. Decision-support summarization only; not an autonomous clinical decision-maker."

    def __init__(self):
        pass

    def validate(self, clinical_report: str, generated_summary: str) -> Dict[str, Any]:
        """
        Validates the generated summary against the source clinical report.
        Returns validation status, detected entities, flagged hallucinations, and disclaimer.
        """
        report_lower = clinical_report.lower()
        summary_lower = generated_summary.lower()

        flags: List[Dict[str, Any]] = []

        # 1. Check for hallucinated drugs
        for drug in KNOWN_DRUGS:
            if drug in summary_lower and drug not in report_lower:
                flags.append({
                    "category": "DRUG_HALLUCINATION",
                    "severity": "HIGH",
                    "entity": drug,
                    "message": f"Summary mentions drug '{drug}' which is absent from source report."
                })

        # 2. Check for hallucinated mutations
        for mut in KNOWN_MUTATIONS:
            if mut in summary_lower and mut not in report_lower:
                # Check if variant parts exist
                parts = mut.split()
                if not any(p in report_lower for p in parts if len(p) > 2):
                    flags.append({
                        "category": "MUTATION_HALLUCINATION",
                        "severity": "HIGH",
                        "entity": mut,
                        "message": f"Summary mentions genomic alteration '{mut}' not documented in source report."
                    })

        # 3. Check for hallucinated adverse events
        for ae in KNOWN_ADVERSE_EVENTS:
            if ae in summary_lower and ae not in report_lower:
                flags.append({
                    "category": "ADVERSE_EVENT_HALLUCINATION",
                    "severity": "MEDIUM",
                    "entity": ae,
                    "message": f"Summary mentions adverse event/toxicity '{ae}' not present in source note."
                })

        # 4. Check for unauthorized clinical recommendations
        for rec in RECOMMENDATION_TRIGGERS:
            if rec in summary_lower and rec not in report_lower:
                flags.append({
                    "category": "UNAUTHORIZED_TREATMENT_RECOMMENDATION",
                    "severity": "CRITICAL",
                    "entity": rec,
                    "message": f"Summary appears to formulate treatment recommendation ('{rec}') beyond factual synthesis."
                })

        # 5. Check for dosage consistency (e.g., numbers followed by mg, mg/day, BID)
        summary_dosages = re.findall(r"\b\d+\s*(?:mg|mg\/day|bid)\b", summary_lower)
        for dose in summary_dosages:
            dose_norm = re.sub(r"\s+", "", dose)
            report_norm = re.sub(r"\s+", "", report_lower)
            if dose_norm not in report_norm:
                flags.append({
                    "category": "DOSAGE_DISCREPANCY",
                    "severity": "HIGH",
                    "entity": dose,
                    "message": f"Summary contains dosage '{dose}' not found in source text."
                })

        # 6. Check for hallucinated symptoms
        for symptom in KNOWN_SYMPTOMS:
            if symptom in summary_lower and symptom not in report_lower:
                flags.append({
                    "category": "SYMPTOM_HALLUCINATION",
                    "severity": "MEDIUM",
                    "entity": symptom,
                    "message": f"Summary mentions symptom '{symptom}' not documented in source report."
                })

        # 7. Check for unauthorized autonomous diagnosis
        for diag in UNAUTHORIZED_DIAGNOSES:
            if diag in summary_lower and diag not in report_lower:
                flags.append({
                    "category": "UNAUTHORIZED_DIAGNOSIS",
                    "severity": "CRITICAL",
                    "entity": diag,
                    "message": f"Summary asserts autonomous diagnostic claim ('{diag}') absent from source."
                })

        is_safe = len([f for f in flags if f["severity"] in ["HIGH", "CRITICAL"]]) == 0

        return {
            "is_safe": is_safe,
            "status": "PASS" if is_safe else "FLAGGED_FOR_REVIEW",
            "flags_count": len(flags),
            "flags": flags,
            "disclaimer": self.DISCLAIMER
        }
