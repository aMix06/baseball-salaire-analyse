"""
EDA with log(Salary) transformation - Baseball Dataset
Reproduces key analyses from baseball_eda_local.py but using log(Salary_1987)
to get correct correlation rankings and scatter plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# SETUP
# ============================================================================

script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / 'eda_log_outputs'
output_dir.mkdir(exist_ok=True)

print(f"Output directory: {output_dir}")

# ============================================================================
# 1. LOAD & CLEAN
# ============================================================================

csv_path = script_dir / 'Baseball.csv'
if not csv_path.exists():
    print(f"ERROR: Could not find Baseball.csv in {script_dir}")
    exit(1)

df = pd.read_csv(csv_path, sep=';')
df['Salary_1987'] = pd.to_numeric(df['Salary_1987'], errors='coerce')

# Drop rows with missing salary — needed for log transform
df_clean = df.dropna(subset=['Salary_1987']).copy()
df_clean['Log_Salary'] = np.log(df_clean['Salary_1987'])

print(f"\nTotal players: {len(df)}")
print(f"Players with salary data: {len(df_clean)} (dropped {len(df) - len(df_clean)} missing)")

# ============================================================================
# 2. COMPARE RAW VS LOG SALARY DISTRIBUTION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Salary_1987 : Brut vs log(Salary)', fontsize=14, fontweight='bold')

# Raw salary
axes[0].hist(df_clean['Salary_1987'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df_clean['Salary_1987'].mean(), color='red', linestyle='--', label=f"Moyenne: {df_clean['Salary_1987'].mean():.1f}")
axes[0].axvline(df_clean['Salary_1987'].median(), color='green', linestyle='--', label=f"Médiane: {df_clean['Salary_1987'].median():.1f}")
axes[0].set_title('Salaire Brut (asymétrique droite)')
axes[0].set_xlabel('Salary_1987 (k$)')
axes[0].set_ylabel('Fréquence')
axes[0].legend()

# Log salary
axes[1].hist(df_clean['Log_Salary'], bins=30, color='darkorange', edgecolor='black', alpha=0.7)
axes[1].axvline(df_clean['Log_Salary'].mean(), color='red', linestyle='--', label=f"Moyenne: {df_clean['Log_Salary'].mean():.2f}")
axes[1].axvline(df_clean['Log_Salary'].median(), color='green', linestyle='--', label=f"Médiane: {df_clean['Log_Salary'].median():.2f}")
axes[1].set_title('log(Salaire) (plus symétrique)')
axes[1].set_xlabel('log(Salary_1987)')
axes[1].set_ylabel('Fréquence')
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / '01_salary_raw_vs_log.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: 01_salary_raw_vs_log.png")
plt.close()

# ============================================================================
# 3. CORRELATION MATRIX WITH LOG(SALARY)
# ============================================================================

key_metrics = ['Hits_86', 'Home_runs_1986', 'Runs_1986', 'Runs_batted_1986',
               'Walks_1986', 'Bat_times_86', 'Longevity', 'Log_Salary']

corr_matrix = df_clean[key_metrics].corr()

print("\n" + "="*60)
print("CORRELATIONS WITH log(Salary_1987) — sorted:")
print("="*60)
salary_corr = corr_matrix['Log_Salary'].drop('Log_Salary').sort_values(ascending=False)
print(salary_corr.round(3))

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True  # upper triangle masked

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax,
    cbar_kws={'label': 'Corrélation'}
)
ax.set_title('Matrice de Corrélation (avec log(Salary))', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '02_correlation_heatmap_log.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 02_correlation_heatmap_log.png")
plt.close()

# ============================================================================
# 4. SCATTER PLOTS: TOP PREDICTORS vs LOG(SALARY)
# ============================================================================

# Use top 6 predictors from the log(salary) correlation ranking
top_predictors = salary_corr.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Variables de Performance vs log(Salary_1987)', fontsize=14, fontweight='bold')

for idx, var in enumerate(top_predictors):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    x = df_clean[var]
    y = df_clean['Log_Salary']

    ax.scatter(x, y, alpha=0.4, s=25, color='steelblue', edgecolors='none')

    # Regression line
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color='red', linewidth=1.5, label=f'R²={r_value**2:.3f}')

    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel('log(Salary_1987)', fontsize=10)
    ax.set_title(f'{var} vs log(Salary)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '03_scatter_vs_log_salary.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 03_scatter_vs_log_salary.png")
plt.close()

# ============================================================================
# 5. COMPARE CORRELATION RANKINGS: RAW vs LOG SALARY
# ============================================================================

# Recompute with raw salary for comparison
key_metrics_raw = ['Hits_86', 'Home_runs_1986', 'Runs_1986', 'Runs_batted_1986',
                   'Walks_1986', 'Bat_times_86', 'Longevity', 'Salary_1987']
corr_raw = df_clean[key_metrics_raw].corr()['Salary_1987'].drop('Salary_1987').sort_values(ascending=False)

comparison = pd.DataFrame({
    'r_raw_salary': corr_raw.round(3),
    'r_log_salary': salary_corr.reindex(corr_raw.index).round(3)
}).sort_values('r_log_salary', ascending=False)

print("\n" + "="*60)
print("RANKING COMPARISON: Raw Salary vs log(Salary)")
print("="*60)
print(comparison.to_string())

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Corrélations avec Salaire : Brut vs log(Salary)', fontsize=13, fontweight='bold')

colors_raw = ['steelblue'] * len(corr_raw)
colors_log = ['darkorange'] * len(salary_corr)

axes[0].barh(corr_raw.index, corr_raw.values, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('vs Salary Brut', fontweight='bold')
axes[0].set_xlabel('Corrélation (r)')
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlim(-0.1, 0.7)

axes[1].barh(salary_corr.index, salary_corr.values, color='darkorange', edgecolor='black', alpha=0.8)
axes[1].set_title('vs log(Salary)', fontweight='bold')
axes[1].set_xlabel('Corrélation (r)')
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlim(-0.1, 0.7)

plt.tight_layout()
plt.savefig(output_dir / '04_correlation_ranking_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 04_correlation_ranking_comparison.png")
plt.close()

# ============================================================================
# 6. EXPORT UPDATED CORRELATION MATRIX
# ============================================================================

corr_matrix.to_csv(output_dir / 'correlation_matrix_log_salary.csv')
print(f"✓ Saved: correlation_matrix_log_salary.csv")

print("\n" + "="*60)
print("EDA LOG COMPLETE")
print("="*60)
print(f"All outputs saved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. 01_salary_raw_vs_log.png       — distribution comparison")
print("  2. 02_correlation_heatmap_log.png  — heatmap with log(salary)")
print("  3. 03_scatter_vs_log_salary.png    — top predictors vs log(salary)")
print("  4. 04_correlation_ranking_comparison.png — raw vs log ranking")
print("  5. correlation_matrix_log_salary.csv")
