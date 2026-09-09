import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

RAW = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_raw_backup.xlsx'
CLN = r'c:\Users\Varun Prasath.J\Desktop\stage 3\oncology_nlp_cleaned.xlsx'

df_raw = pd.read_excel(RAW, dtype=str)
df_cln = pd.read_excel(CLN, dtype=str)

n = len(df_raw)

# ── per-column null stats ──────────────────────────────────────────
cols = df_raw.columns.tolist()
null_before = df_raw.isna().sum()
null_after  = df_cln[cols].isna().sum()

print("="*68)
print(f"{'Column':<25} {'NaN Before':>10} {'% Before':>9} {'NaN After':>10} {'% After':>8}")
print("-"*68)
for c in cols:
    nb = int(null_before[c])
    na = int(null_after[c])
    pb = round(nb/n*100, 2)
    pa = round(na/n*100, 2)
    print(f"{c:<25} {nb:>10} {pb:>8}%  {na:>10} {pa:>7}%")
print("-"*68)
tb = int(null_before.sum())
ta = int(null_after.sum())
print(f"{'TOTAL':<25} {tb:>10} {round(tb/(n*10)*100,2):>8}%  {ta:>10} {round(ta/(n*10)*100,2):>7}%")
print("="*68)

# ── rows with at least 1 null ──────────────────────────────────────
rows_with_null = df_raw.isna().any(axis=1).sum()
print(f"\nRows with >= 1 NaN  (raw)    : {rows_with_null} / {n}  ({round(rows_with_null/n*100,2)}%)")
print(f"Rows with >= 1 NaN  (cleaned): {df_cln[cols].isna().any(axis=1).sum()} / {n}")

# ── rows with multiple nulls ───────────────────────────────────────
null_count_per_row = df_raw.isna().sum(axis=1)
print("\nNull count distribution per row (raw):")
vc = null_count_per_row.value_counts().sort_index()
for k, v in vc.items():
    print(f"  {k} nulls in a row : {v} rows  ({round(v/n*100,2)}%)")

# ── which columns are most often null together ─────────────────────
nullable = ['urgency','gene_mutation','drug_name','dosage_level',
            'adverse_event','symptom_text','annotation_status']
print("\nTop 5 column-pair null co-occurrences:")
pairs = []
for i in range(len(nullable)):
    for j in range(i+1, len(nullable)):
        c1, c2 = nullable[i], nullable[j]
        both = (df_raw[c1].isna() & df_raw[c2].isna()).sum()
        if both > 0:
            pairs.append((c1, c2, both))
pairs.sort(key=lambda x: -x[2])
for c1, c2, cnt in pairs[:5]:
    print(f"  {c1} + {c2} : {cnt} rows both null")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Bar chart: NaN count per column (before vs after)
# ══════════════════════════════════════════════════════════════════
nullable_cols = [c for c in cols if null_before[c] > 0]
nb_vals = [int(null_before[c]) for c in nullable_cols]
na_vals = [int(null_after[c])  for c in nullable_cols]

x = range(len(nullable_cols))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar([i - width/2 for i in x], nb_vals, width,
               color='#E74C3C', label='Before Cleaning', zorder=3)
bars2 = ax.bar([i + width/2 for i in x], na_vals, width,
               color='#2ECC71', label='After Cleaning', zorder=3)

ax.set_xticks(list(x))
ax.set_xticklabels(nullable_cols, rotation=30, ha='right', fontsize=11)
ax.set_ylabel('Number of Null Values', fontsize=12)
ax.set_title('Null Values Per Column — Before vs After Cleaning', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 10, str(h),
                ha='center', va='bottom', fontsize=9, color='#C0392B', fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 10, str(int(h)),
            ha='center', va='bottom', fontsize=9, color='#27AE60', fontweight='bold')

plt.tight_layout()
out1 = r'c:\Users\Varun Prasath.J\Desktop\stage 3\null_bar_chart.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Pie chart: missing vs non-missing in raw data
# ══════════════════════════════════════════════════════════════════
total_cells = n * 10
missing_cells = int(null_before.sum())
present_cells = total_cells - missing_cells

fig2, ax2 = plt.subplots(figsize=(7, 7))
sizes  = [missing_cells, present_cells]
labels = [f'Missing\n{missing_cells:,} cells\n({round(missing_cells/total_cells*100,1)}%)',
          f'Present\n{present_cells:,} cells\n({round(present_cells/total_cells*100,1)}%)']
colors = ['#E74C3C', '#2ECC71']
explode = (0.05, 0)
wedges, texts = ax2.pie(sizes, labels=labels, colors=colors, explode=explode,
                         startangle=140, textprops={'fontsize': 12})
ax2.set_title('Overall Data Completeness (Before Cleaning)\n5,000 rows × 10 columns = 50,000 cells',
              fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
out2 = r'c:\Users\Varun Prasath.J\Desktop\stage 3\null_pie_chart.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Heatmap-style: null distribution across rows
# ══════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(12, 5))
null_matrix = df_raw[nullable].isna().astype(int).values[:500]  # sample 500 rows for viz

im = ax3.imshow(null_matrix.T, aspect='auto', cmap='RdYlGn_r', interpolation='none')
ax3.set_yticks(range(len(nullable)))
ax3.set_yticklabels(nullable, fontsize=10)
ax3.set_xlabel('Patient Record (first 500)', fontsize=11)
ax3.set_title('Null Value Pattern Across First 500 Records\n(Red = Missing, Green = Present)',
              fontsize=13, fontweight='bold', pad=12)
red_patch   = mpatches.Patch(color='#E74C3C', label='Missing (NaN)')
green_patch = mpatches.Patch(color='#2ECC71', label='Present')
ax3.legend(handles=[red_patch, green_patch], loc='upper right', fontsize=10)
plt.tight_layout()
out3 = r'c:\Users\Varun Prasath.J\Desktop\stage 3\null_heatmap.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out3}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Stacked bar: missing% per column
# ══════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(12, 5))
pct_missing = [round(null_before[c]/n*100, 2) for c in nullable_cols]
pct_present = [100 - p for p in pct_missing]

ax4.bar(nullable_cols, pct_present, color='#2ECC71', label='Present %', zorder=3)
ax4.bar(nullable_cols, pct_missing, bottom=pct_present, color='#E74C3C', label='Missing %', zorder=3)
ax4.set_ylabel('Percentage (%)', fontsize=12)
ax4.set_title('Missing vs Present — Percentage per Column (Before Cleaning)',
              fontsize=13, fontweight='bold', pad=12)
ax4.set_xticks(range(len(nullable_cols)))
ax4.set_xticklabels(nullable_cols, rotation=30, ha='right', fontsize=11)
ax4.legend(fontsize=11)
ax4.yaxis.grid(True, linestyle='--', alpha=0.5)
ax4.set_axisbelow(True)
for i, (pm, pp) in enumerate(zip(pct_missing, pct_present)):
    ax4.text(i, pp + pm/2, f'{pm}%', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')
plt.tight_layout()
out4 = r'c:\Users\Varun Prasath.J\Desktop\stage 3\null_stacked_bar.png'
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out4}")

print("\n[DONE] All 4 charts generated.")
