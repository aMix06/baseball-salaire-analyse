import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('Baseball.csv', sep=';')

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df['Salary_1987'] = pd.to_numeric(df['Salary_1987'], errors='coerce')
df_clean = df.dropna(subset=['Salary_1987']).copy()
df_clean['Log_Salary'] = np.log(df_clean['Salary_1987'])

print(f"Rows after dropping missing salary: {len(df_clean)} (dropped {len(df) - len(df_clean)})")
print(f"\nSalary stats:")
print(df_clean['Salary_1987'].describe().round(2))

# ── 3. Model 1 — Raw salary ───────────────────────────────────────────────────
model_raw = smf.ols(
    'Salary_1987 ~ Runs_batted_1986 + Hits_86 + Home_runs_1986 + Longevity',
    data=df_clean
).fit()

print("\n" + "="*60)
print("MODEL 1: Raw Salary")
print("="*60)
print(model_raw.summary())

# ── 4. Model 2 — Log salary ───────────────────────────────────────────────────
model_log = smf.ols(
    'Log_Salary ~ Runs_batted_1986 + Hits_86 + Home_runs_1986 + Longevity',
    data=df_clean
).fit()

print("\n" + "="*60)
print("MODEL 2: Log Salary")
print("="*60)
print(model_log.summary())

# ── 5. Compare R² ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"R² Raw Salary  : {model_raw.rsquared:.4f}")
print(f"R² Log Salary  : {model_log.rsquared:.4f}")
print(f"\nInterpretation: Higher R² = model explains more variance in salary")

# ── 6. Residual plots side by side ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw salary residuals
axes[0].scatter(model_raw.fittedvalues, model_raw.resid, alpha=0.5, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residuals vs Fitted — Raw Salary', fontsize=13)
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')

# Log salary residuals
axes[1].scatter(model_log.fittedvalues, model_log.resid, alpha=0.5, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Residuals vs Fitted — Log Salary', fontsize=13)
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/residual_comparison.png', dpi=150)
plt.show()
print("\nResidual plot saved.")
