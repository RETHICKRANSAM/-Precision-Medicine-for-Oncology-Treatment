import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. Load dataset
# ─────────────────────────────────────────────
FILE_PATH = r"c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_raw_uncleaned_5000x10.xlsx"
OUTPUT_PATH = r"c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_data_quality_report.xlsx"

print("Loading dataset …")
df = pd.read_excel(FILE_PATH, dtype=str)   # read everything as str to catch whitespace/blank issues
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────
def has_leading_trailing(s):
    if isinstance(s, str):
        return s != s.strip()
    return False

def has_extra_internal_spaces(s):
    if isinstance(s, str):
        return '  ' in s
    return False

def is_blank(s):
    if isinstance(s, str):
        return s.strip() == ''
    return False

def safe_nunique(series):
    return series.dropna().nunique()

# ─────────────────────────────────────────────
# 1. DATASET SUMMARY
# ─────────────────────────────────────────────
print("=== DATASET SUMMARY ===")
n_rows, n_cols = df.shape
print(f"Rows: {n_rows}, Columns: {n_cols}")
print(f"Columns: {list(df.columns)}\n")

df_numeric = pd.read_excel(FILE_PATH)   # also keep a numeric-aware copy for dtype info

summary_rows = []
for col in df.columns:
    col_series_str = df[col]
    col_series_num = df_numeric[col]
    missing_count = col_series_str.isna().sum()
    blank_count = col_series_str.apply(is_blank).sum()
    lead_trail_count = col_series_str.apply(has_leading_trailing).sum()
    extra_space_count = col_series_str.apply(has_extra_internal_spaces).sum()
    unique_count = safe_nunique(col_series_str)
    dtype = str(col_series_num.dtype)
    sample_vals = col_series_str.dropna().head(3).tolist()
    summary_rows.append({
        'Column': col,
        'Dtype': dtype,
        'Non-Null Count': n_rows - missing_count,
        'Missing (NaN) Count': missing_count,
        'Blank String Count': blank_count,
        'Leading/Trailing Spaces': lead_trail_count,
        'Extra Internal Spaces': extra_space_count,
        'Unique Values': unique_count,
        'Sample Values': str(sample_vals)
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary[['Column','Dtype','Missing (NaN) Count','Blank String Count','Unique Values']].to_string(index=False))

# ─────────────────────────────────────────────
# 2. MISSING VALUES ANALYSIS
# ─────────────────────────────────────────────
print("\n=== MISSING VALUES ===")
missing_rows = []
for col in df.columns:
    nan_count  = df[col].isna().sum()
    blank_count = df[col].apply(is_blank).sum()
    total_missing = nan_count + blank_count
    pct = round(total_missing / n_rows * 100, 2)
    missing_rows.append({
        'Column': col,
        'NaN Count': nan_count,
        'Blank String Count': blank_count,
        'Total Missing (NaN + Blank)': total_missing,
        'Missing %': pct
    })
df_missing = pd.DataFrame(missing_rows).sort_values('Total Missing (NaN + Blank)', ascending=False)
print(df_missing.to_string(index=False))

# ─────────────────────────────────────────────
# 3. DUPLICATE ANALYSIS
# ─────────────────────────────────────────────
print("\n=== DUPLICATE ANALYSIS ===")
dup_all   = df.duplicated().sum()
dup_keep  = df[df.duplicated(keep=False)]

print(f"Fully duplicate rows: {dup_all}")

dup_summary = [
    {'Metric': 'Total fully-duplicate rows (keep=first)', 'Count': dup_all},
    {'Metric': 'Rows involved in any duplication (keep=False)', 'Count': len(dup_keep)},
]

# Duplicate patient_id
if 'patient_id' in df.columns:
    dup_pid = df['patient_id'].duplicated().sum()
    print(f"Duplicate patient_id values: {dup_pid}")
    dup_summary.append({'Metric': 'Duplicate patient_id values', 'Count': dup_pid})

df_dup_summary = pd.DataFrame(dup_summary)

# Show actual duplicate rows (up to 100)
if len(dup_keep) > 0:
    df_dup_rows = dup_keep.sort_values(by=list(df.columns[:2])).head(100).reset_index(drop=True)
else:
    df_dup_rows = pd.DataFrame(columns=['Info'])
    df_dup_rows.loc[0] = ['No duplicate rows found']

# ─────────────────────────────────────────────
# 4. PATIENT ID VALIDATION
# ─────────────────────────────────────────────
print("\n=== PATIENT ID VALIDATION ===")
pid_issues = []

if 'patient_id' in df.columns:
    pid_series = df['patient_id']
    pid_null    = pid_series.isna().sum()
    pid_blank   = pid_series.apply(is_blank).sum()
    pid_total   = len(pid_series)
    pid_unique  = pid_series.nunique()
    pid_dups    = pid_series.duplicated().sum()

    print(f"Total patient_id values : {pid_total}")
    print(f"Unique patient_id values: {pid_unique}")
    print(f"Duplicate patient_ids   : {pid_dups}")
    print(f"Null patient_ids        : {pid_null}")
    print(f"Blank patient_ids       : {pid_blank}")

    # Uniqueness check
    is_unique = (pid_unique == pid_total - pid_null)
    print(f"Are patient_ids unique?  : {is_unique}")

    # Sequential / ordering check (strip non-numeric prefix)
    try:
        pid_nums = pid_series.dropna().str.extract(r'(\d+)')[0].astype(float)
        pid_sorted = pid_nums.sort_values().reset_index(drop=True)
        expected   = pd.Series(range(int(pid_sorted.min()), int(pid_sorted.min()) + len(pid_sorted)), dtype=float)
        is_sequential = pid_sorted.equals(expected)
        gaps = sorted(set(range(int(pid_sorted.min()), int(pid_sorted.max())+1)) - set(pid_nums.dropna().astype(int).tolist()))
        print(f"Are patient_ids sequential? : {is_sequential}")
        print(f"Gaps in sequence (first 20): {gaps[:20]}")
    except Exception as e:
        is_sequential = f"Could not determine: {e}"
        gaps = []

    # Format consistency
    sample_formats = pid_series.dropna().head(20).tolist()
    format_pattern = Counter(
        re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', str(v))) for v in pid_series.dropna()
    )
    print(f"Format patterns (top 5): {format_pattern.most_common(5)}")

    pid_issues = [
        {'Check': 'Total patient_id values', 'Value': pid_total, 'Status': 'INFO'},
        {'Check': 'Unique patient_id values', 'Value': pid_unique, 'Status': 'INFO'},
        {'Check': 'Duplicate patient_ids', 'Value': pid_dups, 'Status': 'FAIL' if pid_dups > 0 else 'PASS'},
        {'Check': 'Null patient_ids', 'Value': pid_null, 'Status': 'FAIL' if pid_null > 0 else 'PASS'},
        {'Check': 'Blank patient_ids', 'Value': pid_blank, 'Status': 'FAIL' if pid_blank > 0 else 'PASS'},
        {'Check': 'Are patient_ids unique?', 'Value': str(is_unique), 'Status': 'PASS' if is_unique else 'FAIL'},
        {'Check': 'Are patient_ids sequential?', 'Value': str(is_sequential), 'Status': 'PASS' if is_sequential is True else 'FAIL'},
        {'Check': 'Number of gaps in sequence', 'Value': len(gaps), 'Status': 'FAIL' if len(gaps) > 0 else 'PASS'},
        {'Check': 'Sequence gaps (first 20)', 'Value': str(gaps[:20]), 'Status': 'INFO'},
        {'Check': 'Top format patterns', 'Value': str(format_pattern.most_common(5)), 'Status': 'INFO'},
    ]
else:
    pid_issues = [{'Check': 'patient_id column', 'Value': 'NOT FOUND', 'Status': 'FAIL'}]

df_pid = pd.DataFrame(pid_issues)

# ─────────────────────────────────────────────
# 5. CATEGORICAL ANALYSIS
# ─────────────────────────────────────────────
print("\n=== CATEGORICAL ANALYSIS ===")

# Identify likely categorical columns (low cardinality, excluding patient_id and clinical_note)
skip_cols = {'patient_id', 'clinical_note'}
cat_cols = [c for c in df.columns
            if c not in skip_cols and df[c].nunique() <= 50]
# Also include all object dtype cols with <= 200 unique
for c in df.columns:
    if c not in skip_cols and c not in cat_cols and df_numeric[c].dtype == object:
        if df[c].nunique() <= 200:
            cat_cols.append(c)

cat_rows = []
for col in cat_cols:
    val_counts = df[col].value_counts(dropna=False)
    vals_stripped = df[col].dropna().str.strip().str.lower()
    
    # Detect potential inconsistent capitalization
    raw_lower_counts = df[col].dropna().str.lower().value_counts()
    original_case_counts = df[col].dropna().value_counts()
    # Find values that have multiple casing variants
    case_variants = {}
    for v in df[col].dropna():
        key = str(v).strip().lower()
        if key not in case_variants:
            case_variants[key] = set()
        case_variants[key].add(str(v).strip())
    inconsistent_case = {k: list(v) for k, v in case_variants.items() if len(v) > 1}
    
    # Detect leading/trailing whitespace variants
    whitespace_variants = {}
    for v in df[col].dropna():
        key = str(v).strip()
        if key not in whitespace_variants:
            whitespace_variants[key] = set()
        whitespace_variants[key].add(repr(str(v)))
    ws_issues = {k: list(v) for k, v in whitespace_variants.items() if len(v) > 1}

    print(f"\nColumn: {col}")
    print(f"  Unique values ({df[col].nunique()}): {sorted(df[col].dropna().unique().tolist())}")
    if inconsistent_case:
        print(f"  Inconsistent casing: {inconsistent_case}")
    if ws_issues:
        print(f"  Whitespace variants: {ws_issues}")

    for val, cnt in val_counts.items():
        cat_rows.append({
            'Column': col,
            'Value': str(val),
            'Count': cnt,
            'Percentage': round(cnt / n_rows * 100, 2),
            'Has Leading/Trailing Space': has_leading_trailing(str(val)) if val is not np.nan else False,
            'Lower-case Value': str(val).strip().lower() if val is not np.nan else '',
            'Inconsistent Casing Variants': str(inconsistent_case.get(str(val).strip().lower(), [])),
        })

df_cat = pd.DataFrame(cat_rows)

# ─────────────────────────────────────────────
# 6. CLINICAL NOTE QUALITY
# ─────────────────────────────────────────────
print("\n=== CLINICAL NOTE QUALITY ===")

cn_issues_list = []
cn_col = None
for possible in ['clinical_note', 'clinical_notes', 'note', 'notes', 'text']:
    if possible in df.columns:
        cn_col = possible
        break

if cn_col:
    cn = df[cn_col]
    total = len(cn)
    null_count   = cn.isna().sum()
    blank_count  = cn.apply(is_blank).sum()
    valid_notes  = cn.dropna()
    valid_notes  = valid_notes[~valid_notes.apply(is_blank)]

    # Length stats
    lengths = valid_notes.str.len()
    print(f"Clinical note column     : '{cn_col}'")
    print(f"Total notes              : {total}")
    print(f"Null notes               : {null_count}")
    print(f"Blank notes              : {blank_count}")
    print(f"Min length               : {lengths.min()}")
    print(f"Max length               : {lengths.max()}")
    print(f"Mean length              : {round(lengths.mean(), 1)}")
    print(f"Median length            : {lengths.median()}")

    # Noise patterns
    has_html_tags          = valid_notes.str.contains(r'<[^>]+>', regex=True, na=False).sum()
    has_special_chars      = valid_notes.str.contains(r'[^\x00-\x7F]', regex=True, na=False).sum()
    has_multiple_spaces    = valid_notes.str.contains(r'  +', regex=True, na=False).sum()
    has_newlines           = valid_notes.str.contains(r'\n|\r', regex=True, na=False).sum()
    has_tabs               = valid_notes.str.contains(r'\t', regex=True, na=False).sum()
    has_digits_only        = valid_notes.str.match(r'^\d+$', na=False).sum()
    has_short_notes        = (lengths < 20).sum()
    has_very_long_notes    = (lengths > 2000).sum()
    has_placeholder        = valid_notes.str.contains(r'(?i)\bN/A\b|null|none|xxx|test|lorem ipsum|\bna\b', regex=True, na=False).sum()
    has_repeated_chars     = valid_notes.apply(lambda s: bool(re.search(r'(.)\1{4,}', s)) if isinstance(s, str) else False).sum()
    has_all_caps_words     = valid_notes.str.contains(r'\b[A-Z]{4,}\b', regex=True, na=False).sum()
    has_number_patterns    = valid_notes.str.contains(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{10}\b', regex=True, na=False).sum()
    has_email              = valid_notes.str.contains(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', regex=True, na=False).sum()
    has_url                = valid_notes.str.contains(r'http[s]?://', regex=True, na=False).sum()
    has_punct_only         = valid_notes.str.match(r'^[^a-zA-Z0-9]+$', na=False).sum()

    # Commonly misspelled oncology terms (basic check)
    misspell_patterns = {
        'carsinoma (carcinoma)': r'carsinoma',
        'canser (cancer)':       r'canser',
        'tumar (tumor)':         r'tumar',
        'lyphoma (lymphoma)':    r'lyphoma',
        'metatassis (metastasis)': r'metatassis',
        'oncologey (oncology)':  r'oncologey',
        'biopsi (biopsy)':       r'biopsi\b',
        'chemo (informal)':      r'\bchemo\b',
    }
    spell_findings = {}
    for label, pat in misspell_patterns.items():
        cnt = valid_notes.str.contains(pat, case=False, regex=True, na=False).sum()
        if cnt > 0:
            spell_findings[label] = cnt

    print(f"HTML tags found          : {has_html_tags}")
    print(f"Non-ASCII chars          : {has_special_chars}")
    print(f"Multiple spaces          : {has_multiple_spaces}")
    print(f"Newlines/carriage returns: {has_newlines}")
    print(f"Tabs                     : {has_tabs}")
    print(f"Notes < 20 chars         : {has_short_notes}")
    print(f"Notes > 2000 chars       : {has_very_long_notes}")
    print(f"Placeholder-like text    : {has_placeholder}")
    print(f"Repeated chars (noise)   : {has_repeated_chars}")
    print(f"ALL-CAPS words           : {has_all_caps_words}")
    print(f"SSN/phone-like patterns  : {has_number_patterns}")
    print(f"Email addresses          : {has_email}")
    print(f"URLs                     : {has_url}")
    print(f"Punct-only notes         : {has_punct_only}")
    print(f"Possible spelling issues : {spell_findings}")

    cn_rows = [
        {'Metric': 'Clinical note column name', 'Value': cn_col, 'Severity': 'INFO'},
        {'Metric': 'Total notes', 'Value': total, 'Severity': 'INFO'},
        {'Metric': 'Null notes', 'Value': null_count, 'Severity': 'HIGH' if null_count > 0 else 'OK'},
        {'Metric': 'Blank notes', 'Value': blank_count, 'Severity': 'HIGH' if blank_count > 0 else 'OK'},
        {'Metric': 'Min note length (chars)', 'Value': int(lengths.min()), 'Severity': 'INFO'},
        {'Metric': 'Max note length (chars)', 'Value': int(lengths.max()), 'Severity': 'INFO'},
        {'Metric': 'Mean note length (chars)', 'Value': round(lengths.mean(), 1), 'Severity': 'INFO'},
        {'Metric': 'Median note length (chars)', 'Value': lengths.median(), 'Severity': 'INFO'},
        {'Metric': 'Notes with HTML tags', 'Value': has_html_tags, 'Severity': 'HIGH' if has_html_tags > 0 else 'OK'},
        {'Metric': 'Notes with non-ASCII chars', 'Value': has_special_chars, 'Severity': 'MEDIUM' if has_special_chars > 0 else 'OK'},
        {'Metric': 'Notes with multiple spaces', 'Value': has_multiple_spaces, 'Severity': 'MEDIUM' if has_multiple_spaces > 0 else 'OK'},
        {'Metric': 'Notes with newlines/CR', 'Value': has_newlines, 'Severity': 'LOW' if has_newlines > 0 else 'OK'},
        {'Metric': 'Notes with tabs', 'Value': has_tabs, 'Severity': 'LOW' if has_tabs > 0 else 'OK'},
        {'Metric': 'Very short notes (< 20 chars)', 'Value': has_short_notes, 'Severity': 'HIGH' if has_short_notes > 0 else 'OK'},
        {'Metric': 'Very long notes (> 2000 chars)', 'Value': has_very_long_notes, 'Severity': 'LOW' if has_very_long_notes > 0 else 'OK'},
        {'Metric': 'Placeholder-like text', 'Value': has_placeholder, 'Severity': 'HIGH' if has_placeholder > 0 else 'OK'},
        {'Metric': 'Repeated character noise', 'Value': has_repeated_chars, 'Severity': 'MEDIUM' if has_repeated_chars > 0 else 'OK'},
        {'Metric': 'Notes with ALL-CAPS words', 'Value': has_all_caps_words, 'Severity': 'LOW' if has_all_caps_words > 0 else 'OK'},
        {'Metric': 'SSN/phone-like patterns', 'Value': has_number_patterns, 'Severity': 'HIGH' if has_number_patterns > 0 else 'OK'},
        {'Metric': 'Email addresses in notes', 'Value': has_email, 'Severity': 'HIGH' if has_email > 0 else 'OK'},
        {'Metric': 'URLs in notes', 'Value': has_url, 'Severity': 'MEDIUM' if has_url > 0 else 'OK'},
        {'Metric': 'Punctuation-only notes', 'Value': has_punct_only, 'Severity': 'HIGH' if has_punct_only > 0 else 'OK'},
        {'Metric': 'Possible misspellings found', 'Value': str(spell_findings), 'Severity': 'MEDIUM' if spell_findings else 'OK'},
    ]
    df_cn = pd.DataFrame(cn_rows)
else:
    df_cn = pd.DataFrame([{'Metric': 'clinical_note', 'Value': 'Column not found', 'Severity': 'FAIL'}])
    print("WARNING: clinical_note column not found!")

# ─────────────────────────────────────────────
# 7. DATA QUALITY ISSUES (consolidated)
# ─────────────────────────────────────────────
print("\n=== CONSOLIDATING DATA QUALITY ISSUES ===")

dq_rows = []

# Missing/blank per column
for col in df.columns:
    nan_c   = df[col].isna().sum()
    blank_c = df[col].apply(is_blank).sum()
    if nan_c > 0:
        dq_rows.append({'Column': col, 'Issue Type': 'Missing Values (NaN)', 'Count': nan_c, 'Severity': 'HIGH', 'Details': f'{nan_c} NaN values ({round(nan_c/n_rows*100,2)}%)'})
    if blank_c > 0:
        dq_rows.append({'Column': col, 'Issue Type': 'Blank Strings', 'Count': blank_c, 'Severity': 'HIGH', 'Details': f'{blank_c} blank string values'})

# Whitespace
for col in df.columns:
    lt_c = df[col].apply(has_leading_trailing).sum()
    ei_c = df[col].apply(has_extra_internal_spaces).sum()
    if lt_c > 0:
        dq_rows.append({'Column': col, 'Issue Type': 'Leading/Trailing Whitespace', 'Count': lt_c, 'Severity': 'MEDIUM', 'Details': f'{lt_c} values with extra whitespace'})
    if ei_c > 0:
        dq_rows.append({'Column': col, 'Issue Type': 'Extra Internal Spaces', 'Count': ei_c, 'Severity': 'MEDIUM', 'Details': f'{ei_c} values with extra internal spaces'})

# Duplicate rows
if dup_all > 0:
    dq_rows.append({'Column': 'ALL', 'Issue Type': 'Duplicate Rows', 'Count': dup_all, 'Severity': 'HIGH', 'Details': f'{dup_all} fully duplicate rows detected'})

# Duplicate patient_id
if 'patient_id' in df.columns:
    dup_pid_c = df['patient_id'].duplicated().sum()
    if dup_pid_c > 0:
        dq_rows.append({'Column': 'patient_id', 'Issue Type': 'Duplicate patient_id', 'Count': dup_pid_c, 'Severity': 'CRITICAL', 'Details': f'{dup_pid_c} non-unique patient IDs'})

# Inconsistent casing for categorical
for col in cat_cols:
    case_variants_local = {}
    for v in df[col].dropna():
        key = str(v).strip().lower()
        if key not in case_variants_local:
            case_variants_local[key] = set()
        case_variants_local[key].add(str(v).strip())
    inconsistent_local = {k: list(v) for k, v in case_variants_local.items() if len(v) > 1}
    if inconsistent_local:
        dq_rows.append({'Column': col, 'Issue Type': 'Inconsistent Capitalization', 'Count': len(inconsistent_local),
                        'Severity': 'MEDIUM', 'Details': str(dict(list(inconsistent_local.items())[:5]))})

# Sequential gaps in patient_id
if 'patient_id' in df.columns and len(gaps) > 0:
    dq_rows.append({'Column': 'patient_id', 'Issue Type': 'Non-Sequential IDs (Gaps)', 'Count': len(gaps),
                    'Severity': 'MEDIUM', 'Details': f'First 10 gaps: {gaps[:10]}'})

# Clinical note issues
if cn_col:
    for _, row in df_cn.iterrows():
        if row['Severity'] in ('HIGH', 'MEDIUM') and isinstance(row['Value'], (int, float)) and row['Value'] > 0:
            dq_rows.append({'Column': cn_col, 'Issue Type': row['Metric'], 'Count': int(row['Value']),
                            'Severity': row['Severity'], 'Details': f'{int(row["Value"])} notes affected'})

df_dq = pd.DataFrame(dq_rows) if dq_rows else pd.DataFrame(columns=['Column','Issue Type','Count','Severity','Details'])
df_dq = df_dq.sort_values('Severity', key=lambda x: x.map({'CRITICAL':0,'HIGH':1,'MEDIUM':2,'LOW':3,'OK':4,'INFO':5}).fillna(5)).reset_index(drop=True)

print(f"Total data quality issues identified: {len(df_dq)}")
print(df_dq.to_string(index=False))

# ─────────────────────────────────────────────
# 8. WRITE EXCEL REPORT
# ─────────────────────────────────────────────
print(f"\nWriting report to: {OUTPUT_PATH}")

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    # Sheet 1: Dataset Summary
    df_summary.to_excel(writer, sheet_name='Dataset Summary', index=False)

    # Sheet 2: Missing Values
    df_missing.to_excel(writer, sheet_name='Missing Values', index=False)

    # Sheet 3: Duplicate Analysis
    df_dup_summary.to_excel(writer, sheet_name='Duplicate Analysis', index=False, startrow=0)
    if len(dup_keep) > 0:
        df_dup_rows.to_excel(writer, sheet_name='Duplicate Analysis', index=False,
                             startrow=len(df_dup_summary) + 2)

    # Sheet 4: Categorical Analysis
    df_cat.to_excel(writer, sheet_name='Categorical Analysis', index=False)

    # Sheet 5: Patient ID Validation
    df_pid.to_excel(writer, sheet_name='Patient ID Validation', index=False)

    # Sheet 6: Clinical Note Quality
    df_cn.to_excel(writer, sheet_name='Clinical Note Quality', index=False)


    # Sheet 7: Data Quality Issues
    df_dq.to_excel(writer, sheet_name='Data Quality Issues', index=False)

print("\n[OK] Report written successfully!")

# -----------------------------------------------------------------
# 9. FINAL SUMMARY PRINT
# -----------------------------------------------------------------
print("\n" + "="*70)
print("FINAL INSPECTION SUMMARY")
print("="*70)
print(f"Dataset shape           : {n_rows} rows x {n_cols} columns")
print(f"Total DQ issues found   : {len(df_dq)}")
print(f"Fully duplicate rows    : {dup_all}")
if 'patient_id' in df.columns:
    print(f"Duplicate patient_ids   : {df['patient_id'].duplicated().sum()}")
print(f"Columns with NaN        : {(df.isna().sum() > 0).sum()}")
print(f"Columns with blanks     : {sum(df[c].apply(is_blank).sum() > 0 for c in df.columns)}")
print(f"Columns with whitespace : {sum(df[c].apply(has_leading_trailing).sum() > 0 for c in df.columns)}")
if cn_col:
    print(f"Clinical note issues    : HTML={has_html_tags}, Non-ASCII={has_special_chars}, Short={has_short_notes}, Placeholder={has_placeholder}")
print("="*70)
print(f"\nReport saved to: {OUTPUT_PATH}")
