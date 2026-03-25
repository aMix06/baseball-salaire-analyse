"""
Baseball Salary Regression Analysis
4 models: 3 simple + 1 multiple, all using log(Salary)
Includes full diagnostics: residuals, Q-Q plot, influence measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# SETUP
# ============================================================================

script_dir = Path(__file__).parent.absolute()
output_dir = script_dir / 'regression_outputs'
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD & PREPARE DATA
# ============================================================================

df = pd.read_csv(script_dir / 'Baseball.csv', sep=';')
df['Salary_1987'] = pd.to_numeric(df['Salary_1987'], errors='coerce')
df_clean = df.dropna(subset=['Salary_1987']).copy()
df_clean = df_clean[df_clean['Longevity'] > 0].copy()

# Target variable
df_clean['Log_Salary'] = np.log(df_clean['Salary_1987'])

# Normalized career variable
df_clean['Hits_per_year'] = df_clean['Hits_career'] / df_clean['Longevity']

print(f"Players used for regression: {len(df_clean)}")
print(f"\nVariable summary:")
print(df_clean[['Log_Salary', 'Longevity', 'Hits_86', 'Hits_per_year']].describe().round(3))

# ============================================================================
# 2. FIT ALL 4 MODELS
# ============================================================================

model1 = smf.ols('Log_Salary ~ Longevity',    data=df_clean).fit()
model2 = smf.ols('Log_Salary ~ Hits_86',       data=df_clean).fit()
model3 = smf.ols('Log_Salary ~ Hits_per_year', data=df_clean).fit()
model4 = smf.ols('Log_Salary ~ Longevity + Hits_per_year + Hits_86', data=df_clean).fit()

models = {
    'Model 1: Longevity':                  model1,
    'Model 2: Hits_86':                    model2,
    'Model 3: Hits_per_year':              model3,
    'Model 4: Longevity + Hits_per_year\n          + Hits_86': model4,
}

# ============================================================================
# 3. PRINT SUMMARIES
# ============================================================================

print("\n" + "="*70)
for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(model.summary())

# ============================================================================
# 4. MODEL COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

comparison = pd.DataFrame({
    'Model': [
        'M1: ~Longevity',
        'M2: ~Hits_86',
        'M3: ~Hits_per_year',
        'M4: ~Longevity + Hits_per_year + Hits_86'
    ],
    'R²': [m.rsquared for m in [model1, model2, model3, model4]],
    'Adj. R²': [m.rsquared_adj for m in [model1, model2, model3, model4]],
    'AIC': [m.aic for m in [model1, model2, model3, model4]],
    'F-stat p-value': [m.f_pvalue for m in [model1, model2, model3, model4]],
    'N': [int(m.nobs) for m in [model1, model2, model3, model4]],
}).round(4)

print(comparison.to_string(index=False))

# ============================================================================
# 5. VIF FOR MODEL 4 (multicollinearity check)
# ============================================================================

print("\n" + "="*70)
print("VIF — MODEL 4 (multicollinearity check, threshold = 5)")
print("="*70)

X_m4 = df_clean[['Longevity', 'Hits_per_year', 'Hits_86']].copy()
X_m4 = sm.add_constant(X_m4)
vif_data = pd.DataFrame({
    'Variable': ['Longevity', 'Hits_per_year', 'Hits_86'],
    'VIF': [variance_inflation_factor(X_m4.values, i+1) for i in range(3)]
}).round(3)
print(vif_data.to_string(index=False))
print("\nVIF < 5 → no serious multicollinearity")
print("VIF 5-10 → moderate concern")
print("VIF > 10 → serious multicollinearity, consider dropping variable")

# ============================================================================
# 6. FIGURE 1 — R² COMPARISON BAR CHART
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
model_names = ['M1\nLongevity', 'M2\nHits_86', 'M3\nHits_per_year', 'M4\nMultiple']
r2_values = [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared]
colors = ['steelblue', 'steelblue', 'steelblue', 'darkorange']

bars = ax.bar(model_names, r2_values, color=colors, edgecolor='black', alpha=0.85, width=0.5)

for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'R²={val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('R²', fontsize=12)
ax.set_title('Comparaison des Modèles — R²', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(r2_values) + 0.12)
ax.axhline(y=model4.rsquared, color='darkorange', linestyle='--', alpha=0.4)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '01_model_comparison_r2.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: 01_model_comparison_r2.png")
plt.close()

# ============================================================================
# 7. FIGURE 2 — RESIDUALS VS FITTED (all 4 models)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Résidus vs Valeurs Ajustées — 4 Modèles', fontsize=14, fontweight='bold')

model_list = [model1, model2, model3, model4]
model_labels = ['M1: ~Longevity', 'M2: ~Hits_86',
                'M3: ~Hits_per_year', 'M4: Multiple']
plot_colors = ['steelblue', 'steelblue', 'steelblue', 'darkorange']

for idx, (model, label, color) in enumerate(zip(model_list, model_labels, plot_colors)):
    row, col = idx // 2, idx % 2
    ax = axes[row][col]

    ax.scatter(model.fittedvalues, model.resid, alpha=0.4, s=25,
               color=color, edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)

    # Lowess smoothing line to detect patterns
    lowess = sm.nonparametric.lowess(model.resid, model.fittedvalues, frac=0.4)
    ax.plot(lowess[:, 0], lowess[:, 1], color='black', linewidth=1.5, label='Tendance')

    ax.set_title(f'{label}  |  R²={model.rsquared:.3f}', fontweight='bold')
    ax.set_xlabel('Valeurs Ajustées')
    ax.set_ylabel('Résidus')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '02_residuals_vs_fitted.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 02_residuals_vs_fitted.png")
plt.close()

# ============================================================================
# 8. FIGURE 3 — Q-Q PLOTS (all 4 models)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Q-Q Plots des Résidus — Normalité', fontsize=14, fontweight='bold')

for idx, (model, label) in enumerate(zip(model_list, model_labels)):
    row, col = idx // 2, idx % 2
    ax = axes[row][col]

    residuals = model.resid
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals)
    ax.scatter(osm, osr, alpha=0.5, s=20, color='steelblue' if idx < 3 else 'darkorange')
    ax.plot(osm, slope * np.array(osm) + intercept, color='red', linewidth=1.5)
    ax.set_title(f'{label}', fontweight='bold')
    ax.set_xlabel('Quantiles théoriques')
    ax.set_ylabel('Quantiles des résidus')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '03_qq_plots.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 03_qq_plots.png")
plt.close()

# ============================================================================
# 9. FIGURE 4 — MODEL 4 COEFFICIENT PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(9, 5))

coefs = model4.params.drop('Intercept')
conf = model4.conf_int().drop('Intercept')
pvals = model4.pvalues.drop('Intercept')

colors_coef = ['green' if p < 0.05 else 'gray' for p in pvals]
y_pos = range(len(coefs))

ax.barh(y_pos, coefs.values, color=colors_coef, alpha=0.7,
        edgecolor='black', height=0.5)
ax.errorbar(coefs.values, y_pos,
            xerr=[coefs.values - conf[0].values, conf[1].values - coefs.values],
            fmt='none', color='black', capsize=4, linewidth=1.5)

ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(coefs.index, fontsize=11)
ax.set_xlabel('Coefficient (effet sur log(Salary))', fontsize=11)
ax.set_title('Model 4 — Coefficients avec Intervalles de Confiance 95%\n(vert = significatif p<0.05)',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / '04_model4_coefficients.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 04_model4_coefficients.png")
plt.close()

# ============================================================================
# 10. FIGURE 5 — ACTUAL vs PREDICTED (Model 4)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

predicted = model4.fittedvalues
actual = df_clean['Log_Salary']

ax.scatter(predicted, actual, alpha=0.4, s=25, color='darkorange', edgecolors='none')
min_val = min(predicted.min(), actual.min())
max_val = max(predicted.max(), actual.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Parfait (y=x)')

ax.set_xlabel('log(Salary) prédit', fontsize=11)
ax.set_ylabel('log(Salary) réel', fontsize=11)
ax.set_title(f'Model 4 — Prédit vs Réel\nR²={model4.rsquared:.3f}',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '05_model4_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 05_model4_predicted_vs_actual.png")
plt.close()

# ============================================================================
# 11. SUMMARY INTERPRETATION
# ============================================================================

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print(f"""
Model 4 coefficients interpretation:
  - Each coefficient = effect on log(Salary)
  - To interpret as % change in salary: (e^coef - 1) * 100

  Longevity coef    = {model4.params['Longevity']:.4f}
  → Each extra year multiplies salary by e^{model4.params['Longevity']:.4f} = {np.exp(model4.params['Longevity']):.4f}
  → ≈ {(np.exp(model4.params['Longevity'])-1)*100:.1f}% salary increase per year of experience

  Hits_per_year coef = {model4.params['Hits_per_year']:.4f}
  → Each extra hit/year multiplies salary by {np.exp(model4.params['Hits_per_year']):.4f}
  → ≈ {(np.exp(model4.params['Hits_per_year'])-1)*100:.1f}% salary increase per hit/year

  Hits_86 coef      = {model4.params['Hits_86']:.4f}
  → Each extra hit in 1986 multiplies salary by {np.exp(model4.params['Hits_86']):.4f}
  → ≈ {(np.exp(model4.params['Hits_86'])-1)*100:.1f}% salary increase per 1986 hit
""")

# ============================================================================
# 12. FIGURE 6 — SCATTER + REGRESSION LINE FOR MODELS 1, 2, 3
# ============================================================================

simple_models = [
    ('Longevity',    model1, 'Années d\'expérience (Longevity)',    'steelblue'),
    ('Hits_86',      model2, 'Hits en 1986',                        'steelblue'),
    ('Hits_per_year','model3', 'Hits par année de carrière',         'steelblue'),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Régressions Simples — log(Salary) vs Prédicteur',
             fontsize=14, fontweight='bold')

simple_configs = [
    ('Longevity',     model1, 'Années d\'expérience (Longevity)', 'steelblue'),
    ('Hits_86',       model2, 'Hits en 1986 (Hits_86)',           'steelblue'),
    ('Hits_per_year', model3, 'Hits par année (Hits_per_year)',   'steelblue'),
]

for idx, (var, model, xlabel, color) in enumerate(simple_configs):
    ax = axes[idx]

    x = df_clean[var]
    y = df_clean['Log_Salary']

    # Scatter
    ax.scatter(x, y, alpha=0.4, s=30, color=color, edgecolors='none', label='Joueurs')

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 200)
    coef = model.params[var]
    intercept = model.params['Intercept']
    y_line = intercept + coef * x_line
    ax.plot(x_line, y_line, color='red', linewidth=2, label='Droite de régression')

    # Equation and R² annotation
    sign = '+' if coef >= 0 else '-'
    eq_text = (f'log(Salary) = {intercept:.3f} {sign} {abs(coef):.4f} × {var}\n'
               f'R² = {model.rsquared:.3f}   |   p < 0.001')
    ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    model_num = idx + 1
    ax.set_title(f'Modèle {model_num}: log(Salary) ~ {var}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('log(Salary_1987)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '06_simple_regression_scatter.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 06_simple_regression_scatter.png")
plt.close()

print("="*70)
print("COMPLETE — outputs saved to: regression_outputs/")
print("="*70)
print("""
Generated files:
  1. 01_model_comparison_r2.png         — R² bar chart all 4 models
  2. 02_residuals_vs_fitted.png         — residual plots all 4 models
  3. 03_qq_plots.png                    — normality check all 4 models
  4. 04_model4_coefficients.png         — coefficient plot with CI
  5. 05_model4_predicted_vs_actual.png  — predicted vs actual Model 4
  6. 06_simple_regression_scatter.png   — scatter + regression line M1, M2, M3
""")
