import shutil, os, pandas as pd

src  = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_raw_uncleaned_5000x10.xlsx'
dst  = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_raw_backup.xlsx'

# Binary copy — byte-for-byte identical, no Excel re-serialisation
shutil.copy2(src, dst)

# Verify integrity
df_orig   = pd.read_excel(src,  dtype=str)
df_backup = pd.read_excel(dst,  dtype=str)

rows_match = len(df_orig) == len(df_backup)
cols_match = list(df_orig.columns) == list(df_backup.columns)
data_match = df_orig.equals(df_backup)

orig_size   = os.path.getsize(src)
backup_size = os.path.getsize(dst)

print('=== BACKUP VERIFICATION REPORT ===')
print('Source file       :', src)
print('Backup file       :', dst)
print('Source size       :', f'{orig_size:,}', 'bytes')
print('Backup size       :', f'{backup_size:,}', 'bytes')
print('Sizes match       :', orig_size == backup_size)
print('Rows match        :', rows_match, '  (', len(df_orig), 'vs', len(df_backup), ')')
print('Columns match     :', cols_match, '  (', len(df_orig.columns), 'vs', len(df_backup.columns), ')')
print('Data identical    :', data_match)
print('Column names      :', list(df_backup.columns))
print()

pid_col = 'patient_id'
print('First patient_id  (orig)   :', df_orig[pid_col].iloc[0])
print('First patient_id  (backup) :', df_backup[pid_col].iloc[0])
print('Last  patient_id  (orig)   :', df_orig[pid_col].iloc[-1])
print('Last  patient_id  (backup) :', df_backup[pid_col].iloc[-1])
print()

orig_nulls   = df_orig.isna().sum().sum()
backup_nulls = df_backup.isna().sum().sum()
print('Total NaN cells (orig)   :', orig_nulls)
print('Total NaN cells (backup) :', backup_nulls)
print('NaN counts match         :', orig_nulls == backup_nulls)
print()

# Per-column NaN spot check
print('Per-column NaN count comparison:')
for col in df_orig.columns:
    o = df_orig[col].isna().sum()
    b = df_backup[col].isna().sum()
    match = 'OK' if o == b else 'MISMATCH'
    print(f'  {col:<22} orig={o:>4}  backup={b:>4}  [{match}]')

print()
if data_match and rows_match and cols_match and orig_size == backup_size:
    print('[OK] BACKUP IS A PERFECT BYTE-FOR-BYTE COPY. ORIGINAL IS UNTOUCHED.')
else:
    print('[FAIL] Mismatch detected -- review output above.')
