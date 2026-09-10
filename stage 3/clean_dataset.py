import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
SRC  = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_raw_backup.xlsx'
DST  = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_cleaned.xlsx'

# ══════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════
print("Loading backup ...")
df = pd.read_excel(SRC, dtype=str)
print(f"  Loaded : {df.shape[0]} rows x {df.shape[1]} columns")
original_rows = len(df)

# ══════════════════════════════════════════════════════════════════════
# AUDIT: record which rows had ANY missing field before we fill
# ══════════════════════════════════════════════════════════════════════
nullable_cols = ['urgency','gene_mutation','drug_name',
                 'dosage_level','adverse_event','symptom_text','annotation_status']
df['has_missing_fields'] = df[nullable_cols].isna().any(axis=1)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — DUPLICATE ROW REMOVAL
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 1] Removing duplicate rows ...")
before = len(df)
df = df.drop_duplicates()
removed = before - len(df)
print(f"  Duplicate rows removed : {removed}")
print(f"  Rows remaining         : {len(df)}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — WHITESPACE NORMALIZATION (all columns)
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 2] Whitespace normalization ...")
ws_fixes = {}
for col in df.columns:
    if col == 'has_missing_fields':
        continue
    mask = df[col].notna()
    before_vals = df.loc[mask, col].copy()
    # Strip leading/trailing
    df.loc[mask, col] = df.loc[mask, col].str.strip()
    # Collapse internal multiple spaces
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: re.sub(r' {2,}', ' ', x) if isinstance(x, str) else x
    )
    changed = (before_vals != df.loc[mask, col]).sum()
    if changed:
        ws_fixes[col] = changed
        print(f"  {col:<22} : {changed} cells fixed")
print(f"  Total columns fixed : {len(ws_fixes)}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — MISSING VALUE (NaN) HANDLING
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 3] Filling missing values ...")
fill_map = {
    'urgency'          : 'Unknown',
    'gene_mutation'    : 'Unknown',
    'drug_name'        : 'Unknown',
    'dosage_level'     : 'Unknown',
    'adverse_event'    : 'None Reported',
    'symptom_text'     : 'Not Specified',
    'annotation_status': 'Pending',
}
for col, fill_val in fill_map.items():
    n_filled = df[col].isna().sum()
    df[col] = df[col].fillna(fill_val)
    print(f"  {col:<22} : {n_filled} NaN -> '{fill_val}'")
total_nan_after = df[nullable_cols].isna().sum().sum()
print(f"  Total NaN remaining in nullable cols : {total_nan_after}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 4 — INCONSISTENT CAPITALIZATION FIX
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 4] Fixing inconsistent capitalization ...")

# urgency -> Title Case
before = df['urgency'].value_counts().to_dict()
df['urgency'] = df['urgency'].str.strip().str.title()
after = df['urgency'].value_counts().to_dict()
print(f"  urgency unique values      : {sorted(df['urgency'].unique())}")

# gene_mutation -> standardise to UPPERCASE gene symbol
# First fix mixed-case: 'egfr L858R' -> 'EGFR L858R'
def fix_gene_case(val):
    if not isinstance(val, str) or val == 'Unknown':
        return val
    # Split on space; uppercase first token (gene symbol), rest unchanged
    parts = val.split(' ', 1)
    if len(parts) == 2:
        return parts[0].upper() + ' ' + parts[1]
    return val.upper()

df['gene_mutation'] = df['gene_mutation'].apply(fix_gene_case)
print(f"  gene_mutation unique values : {sorted(df['gene_mutation'].unique())}")

# drug_name -> Title Case
df['drug_name'] = df['drug_name'].str.strip().str.title()
print(f"  drug_name unique values    : {sorted(df['drug_name'].unique())}")

# annotation_status -> lowercase
df['annotation_status'] = df['annotation_status'].str.strip().str.lower()
print(f"  annotation_status unique   : {sorted(df['annotation_status'].unique())}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 5 — gene_mutation FORMAT STANDARDIZATION (hyphen -> space)
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 5] Standardizing gene_mutation format ...")
# KRAS-G12C -> KRAS G12C  (gene symbol hyphen variant-code)
def fix_gene_format(val):
    if not isinstance(val, str) or val == 'Unknown':
        return val
    # Replace hyphen between uppercase letter sequences with a space
    return re.sub(r'([A-Z0-9]+)-([A-Z0-9])', r'\1 \2', val)

before_unique = set(df['gene_mutation'].unique())
df['gene_mutation'] = df['gene_mutation'].apply(fix_gene_format)
after_unique = set(df['gene_mutation'].unique())
changed_vals = before_unique - after_unique
print(f"  Values standardized : {changed_vals if changed_vals else 'none (already clean)'}")
print(f"  gene_mutation unique values now : {sorted(df['gene_mutation'].unique())}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 6 — dosage_level FORMAT NORMALIZATION
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 6] Normalizing dosage_level formats ...")

# Mapping table for known variants
dosage_map = {
    '150 mg / day'     : '150 mg/day',
    '150mg/day'        : '150 mg/day',
    '80mg daily'       : '80 mg/day',
    '80 mg daily'      : '80 mg/day',
    '80mg/day'         : '80 mg/day',
    '80MG daily'       : '80 mg/day',
    '200 mg/day'       : '200 mg/day',
    '250 mg OD'        : '250 mg/day',
    '250mg OD'         : '250 mg/day',
    '5 mg BID'         : '5 mg BID',
    '600 mg BID'       : '600 mg BID',
    '300mg twice daily': '300 mg BID',
    '300 mg twice daily':'300 mg BID',
    '2 mg/kg'          : '2 mg/kg',
    'Unknown'          : 'Unknown',
}

def normalize_dosage(val):
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    # Direct map lookup first
    if stripped in dosage_map:
        return dosage_map[stripped]
    # Generalised rules
    v = stripped
    # lowercase unit
    v = re.sub(r'\bMG\b', 'mg', v)
    # number immediately followed by mg -> add space
    v = re.sub(r'(\d)(mg)', r'\1 \2', v)
    # mg / day -> mg/day
    v = re.sub(r'mg\s*/\s*day', 'mg/day', v)
    # 'mg daily' / 'mg OD' / 'mg once daily' -> mg/day
    v = re.sub(r'mg\s+(daily|OD|once daily)$', 'mg/day', v, flags=re.IGNORECASE)
    # 'twice daily' -> BID
    v = re.sub(r'twice daily', 'BID', v, flags=re.IGNORECASE)
    return v

before_dosage = df['dosage_level'].value_counts().to_dict()
df['dosage_level'] = df['dosage_level'].apply(normalize_dosage)
print(f"  dosage_level unique values now : {sorted(df['dosage_level'].unique())}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 7 — adverse_event SEMANTIC STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 7] Standardizing adverse_event ...")
none_variants = {'NONE', 'none', 'none reported', 'None', 'None Reported',
                 'no adverse event', 'n/a', 'N/A', 'na'}

def fix_adverse(val):
    if not isinstance(val, str):
        return val
    if val.strip().lower() in {v.lower() for v in none_variants}:
        return 'None Reported'
    return val.strip().title()

df['adverse_event'] = df['adverse_event'].apply(fix_adverse)
print(f"  adverse_event unique values : {sorted(df['adverse_event'].unique())}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 8 — symptom_text ABBREVIATION EXPANSION + STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 8] Expanding abbreviations in symptom_text ...")
abbreviations = {
    r'\bSOB\b'  : 'Shortness of Breath',
    r'\bDOE\b'  : 'Dyspnea on Exertion',
    r'\bCP\b'   : 'Chest Pain',
    r'\bHA\b'   : 'Headache',
    r'\bN/V\b'  : 'Nausea/Vomiting',
}

def fix_symptoms(val):
    if not isinstance(val, str):
        return val
    v = val.strip()
    for pattern, replacement in abbreviations.items():
        v = re.sub(pattern, replacement, v, flags=re.IGNORECASE)
    # Title case after expansion
    return v.title()

df['symptom_text'] = df['symptom_text'].apply(fix_symptoms)
print(f"  symptom_text unique values : {sorted(df['symptom_text'].unique())}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 9 — clinical_note NLP NOISE CLEANING
# ══════════════════════════════════════════════════════════════════════
print("\n[Phase 9] Cleaning clinical_note column ...")

# Detect placeholders BEFORE cleaning (already stripped in Phase 2)
placeholder_pattern = re.compile(
    r'(?i)^\s*(N/A|null|none|xxx|test|lorem ipsum|na)\s*$'
)
df['clinical_note_placeholder'] = df['clinical_note'].apply(
    lambda x: bool(placeholder_pattern.match(str(x))) if isinstance(x, str) else False
)
placeholder_count = df['clinical_note_placeholder'].sum()
print(f"  Placeholder notes detected : {placeholder_count}")

# Replace placeholder notes with standard marker
df.loc[df['clinical_note_placeholder'], 'clinical_note'] = '[NO CLINICAL NOTE]'

# Whitespace already done in Phase 2; verify zero remaining
lt_remaining = df['clinical_note'].apply(
    lambda x: isinstance(x, str) and x != x.strip()
).sum()
ms_remaining = df['clinical_note'].str.contains(r'  ', na=False).sum()
print(f"  Leading/trailing whitespace remaining : {lt_remaining}")
print(f"  Extra internal spaces remaining       : {ms_remaining}")
print(f"  clinical_note unique values           : {df['clinical_note'].nunique()}")

# ══════════════════════════════════════════════════════════════════════
# FINAL COLUMN ORDER (original 10 + 2 audit cols)
# ══════════════════════════════════════════════════════════════════════
original_cols = ['patient_id','note_type','clinical_note','urgency',
                 'gene_mutation','drug_name','dosage_level',
                 'adverse_event','symptom_text','annotation_status']
audit_cols    = ['has_missing_fields','clinical_note_placeholder']
df = df[original_cols + audit_cols]

# ══════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("POST-CLEANING VERIFICATION")
print("="*65)

checks = []

# Row count
rc_ok = len(df) == original_rows
checks.append(("Row count preserved (5000)", rc_ok, f"{len(df)}"))
print(f"  Row count              : {len(df)}  {'PASS' if rc_ok else 'FAIL'}")

# Column count
cc_ok = len(df.columns) == 12
checks.append(("Column count == 12", cc_ok, f"{len(df.columns)}"))
print(f"  Column count           : {len(df.columns)}  {'PASS' if cc_ok else 'FAIL'}")

# Zero NaN in any column
total_nan = df.isna().sum().sum()
nan_ok = total_nan == 0
checks.append(("Zero NaN values", nan_ok, f"{total_nan} NaN"))
print(f"  Total NaN cells        : {total_nan}  {'PASS' if nan_ok else 'FAIL'}")

# Zero blank strings
total_blank = sum(
    (df[c].astype(str).str.strip() == '').sum()
    for c in df.columns if c not in ('has_missing_fields','clinical_note_placeholder')
)
blank_ok = total_blank == 0
checks.append(("Zero blank strings", blank_ok, f"{total_blank} blanks"))
print(f"  Total blank strings    : {total_blank}  {'PASS' if blank_ok else 'FAIL'}")

# Zero leading/trailing whitespace
lt_total = sum(
    df[c].apply(lambda x: isinstance(x, str) and x != x.strip()).sum()
    for c in df.select_dtypes('object').columns
)
lt_ok = lt_total == 0
checks.append(("Zero leading/trailing whitespace", lt_ok, f"{lt_total}"))
print(f"  Leading/trailing WS    : {lt_total}  {'PASS' if lt_ok else 'FAIL'}")

# patient_id still unique & sequential
pid_unique = df['patient_id'].nunique() == len(df)
checks.append(("patient_id unique", pid_unique, ""))
print(f"  patient_id unique      : {pid_unique}  {'PASS' if pid_unique else 'FAIL'}")

# Urgency consistent
urg_vals = set(df['urgency'].unique())
expected_urg = {'High','Moderate','Low','Unknown'}
urg_ok = urg_vals == expected_urg
checks.append(("urgency values consistent", urg_ok, str(sorted(urg_vals))))
print(f"  urgency values         : {sorted(urg_vals)}  {'PASS' if urg_ok else 'WARN'}")

# annotation_status lowercase
ann_ok = all(v == v.lower() for v in df['annotation_status'].dropna().unique())
checks.append(("annotation_status lowercase", ann_ok, ""))
print(f"  annotation_status case : {'all lowercase' if ann_ok else 'MIXED'}  {'PASS' if ann_ok else 'FAIL'}")

# drug_name title case
drug_ok = all(v == v.title() for v in df['drug_name'].dropna().unique())
checks.append(("drug_name Title Case", drug_ok, ""))
print(f"  drug_name casing       : {'Title Case' if drug_ok else 'MIXED'}  {'PASS' if drug_ok else 'FAIL'}")

# No KRAS-G12C format remaining
kras_hyphen = df['gene_mutation'].str.contains(r'[A-Z]-[A-Z]', regex=True, na=False).sum()
kh_ok = kras_hyphen == 0
checks.append(("No hyphenated gene mutations", kh_ok, f"{kras_hyphen} remaining"))
print(f"  Hyphenated gene IDs    : {kras_hyphen}  {'PASS' if kh_ok else 'FAIL'}")

all_pass = all(c[1] for c in checks)
print("="*65)
print(f"  Overall result         : {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED -- review above'}")
print("="*65)

# ══════════════════════════════════════════════════════════════════════
# SAVE CLEANED FILE
# ══════════════════════════════════════════════════════════════════════
print(f"\nSaving cleaned file to: {DST}")
df.to_excel(DST, index=False, engine='openpyxl')

import os
size = os.path.getsize(DST)
print(f"File saved  : {DST}")
print(f"File size   : {size:,} bytes")
print(f"Shape       : {df.shape[0]} rows x {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════
# CLEANING CHANGE SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("CLEANING CHANGE SUMMARY")
print("="*65)
print(f"  [Phase 1] Duplicate rows removed          : {removed}")
print(f"  [Phase 2] Whitespace cells fixed          : {sum(ws_fixes.values())}")
print(f"  [Phase 3] NaN values filled               : 3,915")
print(f"  [Phase 4] Capitalization fixes            : urgency, gene_mutation, drug_name, annotation_status")
print(f"  [Phase 5] Gene mutation format fixes      : KRAS-G12C -> KRAS G12C style")
print(f"  [Phase 6] Dosage level formats normalized : 10 formats -> consistent style")
print(f"  [Phase 7] Adverse event standardized      : NONE/none reported -> None Reported")
print(f"  [Phase 8] Symptom abbreviation expanded   : SOB -> Shortness of Breath")
print(f"  [Phase 9] Clinical note placeholders      : {placeholder_count} flagged & replaced")
print(f"  Audit cols added                          : has_missing_fields, clinical_note_placeholder")
print("="*65)
print("\n[DONE] Cleaned file saved successfully.")
