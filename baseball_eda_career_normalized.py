"""
EDA - Normalized Career Stats vs log(Salary)
Creates Hits_career/Longevity, Runs_batted_career/Longevity, etc.
and explores their correlation and scatter plots against log(Salary)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# SETUP
# ============================================================================

script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / 'eda_career_normalized'
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(script_dir / 'Baseball.csv', sep=';')
df['Salary_1987'] = pd.to_numeric(df['Salary_1987'], errors='coerce')
df_clean = df.dropna(subset=['Salary_1987']).copy()

# ============================================================================
# 1. CREATE NORMALIZED CAREER VARIABLES
# ============================================================================

# Avoid division by zero — any player with Longevity=0 gets NaN
df_clean = df_clean[df_clean['Longevity'] > 0].copy()

df_clean['Log_Salary']              = np.log(df_clean['Salary_1987'])
df_clean['Hits_per_year']           = df_clean['Hits_career'] / df_clean['Longevity']
df_clean['HomeRuns_per_year']       = df_clean['Home_runs_career'] / df_clean['Longevity']
df_clean['Runs_per_year']           = df_clean['Runs_career'] / df_clean['Longevity']
df_clean['RBI_per_year']            = df_clean['Runs_batted_career'] / df_clean['Longevity']
df_clean['Walks_per_year']          = df_clean['Walks_career'] / df_clean['Longevity']
df_clean['BatTimes_per_year']       = df_clean['Bat_times_career'] / df_clean['Longevity']

normalized_vars = [
    'Hits_per_year',
    'HomeRuns_per_year',
    'Runs_per_year',
    'RBI_per_year',
    'Walks_per_year',
    'BatTimes_per_year',
]

print(f"Players after cleaning: {len(df_clean)}")

# ============================================================================
# 2. CORRELATION TABLE: NORMALIZED CAREER STATS vs LOG(SALARY)
# ============================================================================

print("\n" + "="*60)
print("CORRELATIONS WITH log(Salary) — Normalized Career Stats")
print("="*60)

corr_results = {}
for var in normalized_vars:
    r, p = stats.pearsonr(df_clean[var], df_clean['Log_Salary'])
    corr_results[var] = {'r': round(r, 3), 'p_value': round(p, 4), 'R2': round(r**2, 3)}

corr_df = pd.DataFrame(corr_results).T.sort_values('r', ascending=False)
print(corr_df.to_string())

# Also compare to raw career stats
print("\n" + "="*60)
print("COMPARISON: Raw career stats vs Normalized career stats")
print("="*60)

raw_career = ['Hits_career', 'Home_runs_career', 'Runs_career',
              'Runs_batted_career', 'Walks_career', 'Bat_times_career']

comparison_rows = []
for raw, norm in zip(raw_career, normalized_vars):
    r_raw, _ = stats.pearsonr(df_clean[raw], df_clean['Log_Salary'])
    r_norm, _ = stats.pearsonr(df_clean[norm], df_clean['Log_Salary'])
    comparison_rows.append({
        'Raw variable': raw,
        'r (raw)': round(r_raw, 3),
        'Normalized variable': norm,
        'r (normalized)': round(r_norm, 3),
        'Change': round(r_norm - r_raw, 3)
    })

comp_df = pd.DataFrame(comparison_rows)
print(comp_df.to_string(index=False))

# ============================================================================
# 3. SCATTER PLOTS: NORMALIZED CAREER STATS vs LOG(SALARY)
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Stats Carrière Normalisées (÷ Longevity) vs log(Salary_1987)',
             fontsize=14, fontweight='bold')

for idx, var in enumerate(normalized_vars):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    x = df_clean[var]
    y = df_clean['Log_Salary']

    ax.scatter(x, y, alpha=0.4, s=25, color='darkorange', edgecolors='none')

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='red', linewidth=1.5)

    sig = "✓ significatif" if p_value < 0.05 else "✗ non-significatif"
    ax.set_title(f'{var}\nr={r_value:.3f}, R²={r_value**2:.3f} ({sig})', fontsize=9.5)
    ax.set_xlabel(var, fontsize=9)
    ax.set_ylabel('log(Salary)', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_scatter_normalized_career_vs_log_salary.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: 01_scatter_normalized_career_vs_log_salary.png")
plt.close()

# ============================================================================
# 4. CORRELATION HEATMAP: NORMALIZED + SEASON + LOG(SALARY)
# ============================================================================

season_vars = ['Hits_86', 'Runs_batted_1986', 'Walks_1986', 'Home_runs_1986']
all_vars = normalized_vars + season_vars + ['Longevity', 'Log_Salary']

corr_full = df_clean[all_vars].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(corr_full, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    corr_full, mask=mask,
    annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    cbar_kws={'label': 'Corrélation'}
)
ax.set_title('Corrélations : Stats Normalisées + Saison 1986 + log(Salary)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '02_heatmap_full_normalized.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 02_heatmap_full_normalized.png")
plt.close()

# ============================================================================
# 5. BAR CHART: ALL PREDICTORS RANKED BY CORRELATION WITH LOG(SALARY)
# ============================================================================

all_predictors = normalized_vars + season_vars + ['Longevity']
corr_ranking = {}
for var in all_predictors:
    r, _ = stats.pearsonr(df_clean[var], df_clean['Log_Salary'])
    corr_ranking[var] = round(r, 3)

ranking_series = pd.Series(corr_ranking).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['darkorange' if '_per_year' in v else
          'steelblue' if '_86' in v or v == 'Longevity' else 'gray'
          for v in ranking_series.index]

bars = ax.barh(ranking_series.index, ranking_series.values,
               color=colors, edgecolor='black', alpha=0.85)

ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Corrélation avec log(Salary_1987)', fontsize=11)
ax.set_title('Tous les Prédicteurs Classés par Corrélation avec log(Salary)',
             fontsize=12, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkorange', label='Career normalisé (÷ Longevity)'),
    Patch(facecolor='steelblue', label='Saison 1986 + Longevity'),
]
ax.legend(handles=legend_elements, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / '03_all_predictors_ranked.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 03_all_predictors_ranked.png")
plt.close()

print("\n" + "="*60)
print("COMPLETE — outputs saved to: eda_career_normalized/")
print("="*60)
print("\nGenerated files:")
print("  1. 01_scatter_normalized_career_vs_log_salary.png")
print("  2. 02_heatmap_full_normalized.png")
print("  3. 03_all_predictors_ranked.png")
