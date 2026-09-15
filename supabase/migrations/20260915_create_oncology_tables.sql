-- Migration: create_oncology_precision_medicine_tables
-- Target: Supabase Postgres
-- Project: Precision Medicine for Oncology Treatment Optimization

-- 1. Patients Table
CREATE TABLE IF NOT EXISTS public.patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER,
    sex TEXT,
    patient_group TEXT,
    cancer_type TEXT NOT NULL,
    cancer_stage INTEGER,
    gene_mutation TEXT,
    comorbidities TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Clinical Biomarkers & Vitals Table
CREATE TABLE IF NOT EXISTS public.clinical_biomarkers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id TEXT REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    encounter_date DATE,
    egfr_expression NUMERIC,
    kras_expression NUMERIC,
    alk_expression NUMERIC,
    ctdna_level NUMERIC,
    tumor_marker NUMERIC,
    wbc_count NUMERIC,
    hemoglobin NUMERIC,
    platelet_count NUMERIC,
    creatinine NUMERIC,
    bilirubin NUMERIC,
    alt NUMERIC,
    ast NUMERIC,
    heart_rate NUMERIC,
    systolic_bp NUMERIC,
    oxygen_saturation NUMERIC,
    temperature NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Treatments & Clinical Triage Table
CREATE TABLE IF NOT EXISTS public.treatments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id TEXT REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    treatment_drug TEXT NOT NULL,
    dosage_mg NUMERIC,
    dosage_level TEXT,
    treatment_adherence_pct NUMERIC,
    prior_therapies INTEGER,
    treatment_response TEXT,
    toxicity_score INTEGER,
    toxicity_risk TEXT,
    risk_score NUMERIC,
    reported_symptoms TEXT,
    reported_adverse_events TEXT,
    urgency TEXT,
    clinical_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Compound Scenarios Table (GenAI Stage)
CREATE TABLE IF NOT EXISTS public.compound_scenarios (
    scenario_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES public.patients(patient_id) ON DELETE SET NULL,
    severity TEXT NOT NULL CHECK (severity IN ('Mild', 'Moderate', 'Severe', 'Wildcard')),
    patient_scenario TEXT NOT NULL,
    compound_interactions JSONB NOT NULL DEFAULT '[]'::jsonb,
    potential_risk_context JSONB NOT NULL DEFAULT '[]'::jsonb,
    seed_conditions JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_name TEXT NOT NULL DEFAULT 'Qwen/Qwen2.5-0.5B-Instruct',
    prompt_version TEXT NOT NULL DEFAULT 'v1.1',
    preservation_rate NUMERIC NOT NULL DEFAULT 1.0,
    validation_status TEXT NOT NULL DEFAULT 'passed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_patients_cancer_type ON public.patients(cancer_type);
CREATE INDEX IF NOT EXISTS idx_patients_gene_mutation ON public.patients(gene_mutation);
CREATE INDEX IF NOT EXISTS idx_biomarkers_patient_id ON public.clinical_biomarkers(patient_id);
CREATE INDEX IF NOT EXISTS idx_treatments_patient_id ON public.treatments(patient_id);
CREATE INDEX IF NOT EXISTS idx_treatments_urgency ON public.treatments(urgency);
CREATE INDEX IF NOT EXISTS idx_scenarios_severity ON public.compound_scenarios(severity);
CREATE INDEX IF NOT EXISTS idx_scenarios_patient_id ON public.compound_scenarios(patient_id);

-- Enable Row Level Security (RLS)
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.clinical_biomarkers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.treatments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.compound_scenarios ENABLE ROW LEVEL SECURITY;

-- Permissive RLS Policies for Anon & Authenticated access
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow read access on patients' AND tablename = 'patients') THEN
        CREATE POLICY "Allow read access on patients" ON public.patients FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow insert on patients' AND tablename = 'patients') THEN
        CREATE POLICY "Allow insert on patients" ON public.patients FOR INSERT WITH CHECK (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow read access on clinical_biomarkers' AND tablename = 'clinical_biomarkers') THEN
        CREATE POLICY "Allow read access on clinical_biomarkers" ON public.clinical_biomarkers FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow insert on clinical_biomarkers' AND tablename = 'clinical_biomarkers') THEN
        CREATE POLICY "Allow insert on clinical_biomarkers" ON public.clinical_biomarkers FOR INSERT WITH CHECK (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow read access on treatments' AND tablename = 'treatments') THEN
        CREATE POLICY "Allow read access on treatments" ON public.treatments FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow insert on treatments' AND tablename = 'treatments') THEN
        CREATE POLICY "Allow insert on treatments" ON public.treatments FOR INSERT WITH CHECK (true);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow read access on compound_scenarios' AND tablename = 'compound_scenarios') THEN
        CREATE POLICY "Allow read access on compound_scenarios" ON public.compound_scenarios FOR SELECT USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'Allow insert on compound_scenarios' AND tablename = 'compound_scenarios') THEN
        CREATE POLICY "Allow insert on compound_scenarios" ON public.compound_scenarios FOR INSERT WITH CHECK (true);
    END IF;
END $$;
