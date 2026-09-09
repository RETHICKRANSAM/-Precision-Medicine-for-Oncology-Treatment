import pandas as pd
import numpy as np

RAW  = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_raw_backup.xlsx'
CLN  = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_cleaned.xlsx'

df_raw = pd.read_excel(RAW, dtype=str)
df_cln = pd.read_excel(CLN, dtype=str)

print('=== SHAPE ===')
print('Before:', df_raw.shape)
print('After :', df_cln.shape)
print('Rows added/removed:', len(df_cln) - len(df_raw))
print('Columns added     :', len(df_cln.columns) - len(df_raw.columns))

print()
print('=== DUPLICATES ===')
print('Duplicate rows BEFORE :', df_raw.duplicated().sum())
print('Duplicate rows AFTER  :', df_cln.duplicated().sum())

print()
print('=== MISSING VALUES PER COLUMN ===')
header = f"{'Column':<25} {'Before NaN':>12} {'After NaN':>12} {'Filled':>8}"
print(header)
print('-' * 60)
for col in df_raw.columns:
    before = int(df_raw[col].isna().sum())
    after  = int(df_cln[col].isna().sum()) if col in df_cln.columns else 0
    filled = before - after
    print(f"{col:<25} {before:>12} {after:>12} {filled:>8}")

total_before = int(df_raw.isna().sum().sum())
total_after  = int(df_cln[df_raw.columns].isna().sum().sum())
print(f"{'TOTAL':<25} {total_before:>12} {total_after:>12} {total_before - total_after:>8}")

print()
print('=== UNIQUE VALUES PER COLUMN (before vs after) ===')
skip = {'patient_id', 'clinical_note'}
for col in df_raw.columns:
    if col in skip:
        continue
    b = df_raw[col].nunique(dropna=True)
    a = df_cln[col].nunique(dropna=True) if col in df_cln.columns else 0
    print(f"  {col:<25} before={b:>3}  after={a:>3}")

print()
print('=== WHITESPACE IN clinical_note ===')
lt_before = df_raw['clinical_note'].apply(lambda x: isinstance(x, str) and x != x.strip()).sum()
lt_after  = df_cln['clinical_note'].apply(lambda x: isinstance(x, str) and x != x.strip()).sum()
ms_before = df_raw['clinical_note'].str.contains('  ', na=False).sum()
ms_after  = df_cln['clinical_note'].str.contains('  ', na=False).sum()
print('Leading/trailing spaces: before =', lt_before, ' after =', lt_after)
print('Extra internal spaces  : before =', ms_before, ' after =', ms_after)

print()
print('=== AUDIT COLUMNS IN CLEANED FILE ===')
hm = (df_cln['has_missing_fields'].astype(str).str.lower() == 'true').sum()
cp = (df_cln['clinical_note_placeholder'].astype(str).str.lower() == 'true').sum()
print('has_missing_fields = True        :', hm, 'rows (originally had NaN in at least 1 field)')
print('clinical_note_placeholder = True :', cp, 'rows')

print()
print('=== CATEGORICAL VALUES: BEFORE vs AFTER ===')
cat_cols = ['urgency', 'gene_mutation', 'drug_name', 'dosage_level',
            'adverse_event', 'symptom_text', 'annotation_status']
for col in cat_cols:
    print()
    print('  Column:', col)
    bvals = sorted([str(v) for v in df_raw[col].dropna().unique()])
    avals = sorted([str(v) for v in df_cln[col].dropna().unique()])
    print('    BEFORE:', bvals)
    print('    AFTER :', avals)

print()
print('=== DONE ===')
