"""
Exploratory Data Analysis (EDA) - Baseball Dataset
Analyse des données salariales de joueurs de baseball

This script performs comprehensive EDA including:
- Indicateurs de tendance centrale (central tendency measures)
- Indicateurs de dispersion (dispersion measures)
- Effectif de chaque classe (class frequencies)
- Graphiques (scatter plots, histograms, etc.)

USAGE:
1. Place this script in the same folder as your Baseball.csv file
2. Run: python baseball_eda.py
3. All outputs will be saved in a new folder called 'eda_outputs'
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

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# SETUP - Create output directory
# ============================================================================

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Create output directory
output_dir = script_dir / 'eda_outputs'
output_dir.mkdir(exist_ok=True)

print(f"Script directory: {script_dir}")
print(f"Output directory: {output_dir}")

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS - BASEBALL DATASET")
print("="*80)
print("\n1. DATA LOADING AND INITIAL EXPLORATION\n")

# Try to find the Baseball.csv file
csv_path = script_dir / 'Baseball.csv'
if not csv_path.exists():
    print(f"ERROR: Could not find Baseball.csv in {script_dir}")
    print("Please make sure Baseball.csv is in the same folder as this script.")
    exit(1)

# Load the dataset
df = pd.read_csv(csv_path, sep=';')

print(f"Dataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst few rows:")
print(df.head())

print(f"\n\nDataset info:")
print(df.info())

print(f"\n\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================

print("\n" + "="*80)
print("2. DATA CLEANING")
print("="*80 + "\n")

# Check for missing values
print("Missing values by column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0])

# Handle Salary_1987 - convert to numeric (NA should become NaN)
print(f"\n\nOriginal Salary_1987 type: {df['Salary_1987'].dtype}")
print(f"Unique non-numeric values in Salary_1987: {df['Salary_1987'][pd.to_numeric(df['Salary_1987'], errors='coerce').isna()].unique()}")

df['Salary_1987'] = pd.to_numeric(df['Salary_1987'], errors='coerce')
print(f"After conversion - Missing salaries: {df['Salary_1987'].isna().sum()} ({df['Salary_1987'].isna().sum()/len(df)*100:.1f}%)")

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

# ============================================================================
# 3. INDICATEURS DE TENDANCE CENTRALE (CENTRAL TENDENCY)
# ============================================================================

print("\n" + "="*80)
print("3. INDICATEURS DE TENDANCE CENTRALE")
print("="*80 + "\n")

central_tendency = pd.DataFrame({
    'Mean': df[numeric_cols].mean(),
    'Median': df[numeric_cols].median(),
    'Mode': df[numeric_cols].mode().iloc[0] if len(df[numeric_cols].mode()) > 0 else np.nan
})

print("Statistical Summary (Mean, Median, Mode):")
print(central_tendency.round(2))

# Additional statistics
print("\n\nComplete Descriptive Statistics:")
print(df[numeric_cols].describe().round(2))

# ============================================================================
# 4. INDICATEURS DE DISPERSION (DISPERSION MEASURES)
# ============================================================================

print("\n" + "="*80)
print("4. INDICATEURS DE DISPERSION")
print("="*80 + "\n")

dispersion = pd.DataFrame({
    'Std_Dev': df[numeric_cols].std(),
    'Variance': df[numeric_cols].var(),
    'Range': df[numeric_cols].max() - df[numeric_cols].min(),
    'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
    'CV': (df[numeric_cols].std() / df[numeric_cols].mean()) * 100  # Coefficient of variation
})

print("Dispersion Measures:")
print(dispersion.round(2))

# ============================================================================
# 5. EFFECTIF DE CHAQUE CLASSE (CLASS FREQUENCIES)
# ============================================================================

print("\n" + "="*80)
print("5. EFFECTIF DE CHAQUE CLASSE DES VARIABLES CATÉGORIELLES")
print("="*80 + "\n")

for col in categorical_cols:
    print(f"\n{col}:")
    print("-" * 60)
    freq = df[col].value_counts()
    freq_pct = (freq / len(df)) * 100
    freq_df = pd.DataFrame({
        'Count': freq,
        'Percentage': freq_pct
    })
    print(freq_df)
    print(f"Total unique values: {df[col].nunique()}")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("6. CORRELATION ANALYSIS")
print("="*80 + "\n")

# Select key performance metrics for correlation
key_metrics = ['Hits_86', 'Home_runs_1986', 'Runs_1986', 'Runs_batted_1986', 
               'Walks_1986', 'Bat_times_86', 'Salary_1987', 'Longevity']
correlation_df = df[key_metrics].corr()

print("Correlation Matrix (Key Variables):")
print(correlation_df.round(3))

print("\n\nStrongest correlations with Salary_1987:")
salary_corr = correlation_df['Salary_1987'].sort_values(ascending=False)
print(salary_corr[salary_corr.index != 'Salary_1987'])

# ============================================================================
# 7. OUTLIER DETECTION
# ============================================================================

print("\n" + "="*80)
print("7. OUTLIER DETECTION (IQR Method)")
print("="*80 + "\n")

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

print("Outlier Analysis:")
outlier_summary = []
for col in numeric_cols:
    if df[col].notna().sum() > 0:
        n_outliers, lower, upper = detect_outliers_iqr(df, col)
        outlier_summary.append({
            'Variable': col,
            'N_Outliers': n_outliers,
            'Pct_Outliers': (n_outliers / len(df)) * 100,
            'Lower_Bound': lower,
            'Upper_Bound': upper
        })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("8. GENERATING VISUALIZATIONS")
print("="*80 + "\n")

# ---- Figure 1: Distribution of Key Variables ----
fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
fig1.suptitle('Distribution des Variables Clés (1986)', fontsize=16, fontweight='bold')

variables_to_plot = ['Hits_86', 'Home_runs_1986', 'Runs_1986', 
                     'Runs_batted_1986', 'Walks_1986', 'Salary_1987']

for idx, var in enumerate(variables_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    data = df[var].dropna()
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(var)
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution: {var}')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / '01_distributions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '01_distributions.png'}")
plt.close()

# ---- Figure 2: Box Plots for Outlier Visualization ----
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('Box Plots - Détection des Valeurs Aberrantes', fontsize=16, fontweight='bold')

for idx, var in enumerate(variables_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    data = df[var].dropna()
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    
    ax.set_ylabel(var)
    ax.set_title(f'Box Plot: {var}')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '02_boxplots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '02_boxplots.png'}")
plt.close()

# ---- Figure 3: Correlation Heatmap ----
fig3, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_df, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation - Variables Clés', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / '03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '03_correlation_heatmap.png'}")
plt.close()

# ---- Figure 4: Scatter Plots - Performance vs Salary ----
fig4, axes = plt.subplots(2, 2, figsize=(14, 12))
fig4.suptitle('Nuages de Points - Performance vs Salaire 1987', fontsize=16, fontweight='bold')

performance_vars = ['Hits_86', 'Home_runs_1986', 'Runs_batted_1986', 'Longevity']
colors = ['blue', 'green', 'orange', 'purple']

for idx, (var, color) in enumerate(zip(performance_vars, colors)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Remove NaN values
    mask = df[var].notna() & df['Salary_1987'].notna()
    x = df.loc[mask, var]
    y = df.loc[mask, 'Salary_1987']
    
    ax.scatter(x, y, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    if len(x) > 0:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Regression line')
        
        # Calculate R²
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(var, fontsize=11)
    ax.set_ylabel('Salary_1987 ($ thousands)', fontsize=11)
    ax.set_title(f'{var} vs Salary', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(output_dir / '04_scatter_performance_salary.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '04_scatter_performance_salary.png'}")
plt.close()

# ---- Figure 5: League and Division Analysis ----
fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
fig5.suptitle('Analyse par Ligue et Division', fontsize=16, fontweight='bold')

# League distribution
ax1 = axes[0, 0]
league_counts = df['League_1986'].value_counts()
ax1.bar(league_counts.index, league_counts.values, color=['steelblue', 'coral'], edgecolor='black')
ax1.set_xlabel('League')
ax1.set_ylabel('Number of Players')
ax1.set_title('Distribution des Joueurs par Ligue')
ax1.grid(True, alpha=0.3, axis='y')

# Division distribution
ax2 = axes[0, 1]
division_counts = df['Division_1986'].value_counts()
ax2.bar(division_counts.index, division_counts.values, color=['lightgreen', 'lightyellow'], edgecolor='black')
ax2.set_xlabel('Division')
ax2.set_ylabel('Number of Players')
ax2.set_title('Distribution des Joueurs par Division')
ax2.grid(True, alpha=0.3, axis='y')

# Salary by League
ax3 = axes[1, 0]
df.boxplot(column='Salary_1987', by='League_1986', ax=ax3)
ax3.set_xlabel('League')
ax3.set_ylabel('Salary_1987 ($ thousands)')
ax3.set_title('Salaire par Ligue')
plt.sca(ax3)
plt.xticks(rotation=0)

# Salary by Division
ax4 = axes[1, 1]
df.boxplot(column='Salary_1987', by='Division_1986', ax=ax4)
ax4.set_xlabel('Division')
ax4.set_ylabel('Salary_1987 ($ thousands)')
ax4.set_title('Salaire par Division')
plt.sca(ax4)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(output_dir / '05_league_division_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '05_league_division_analysis.png'}")
plt.close()

# ---- Figure 6: Position Analysis ----
fig6, axes = plt.subplots(2, 1, figsize=(14, 10))
fig6.suptitle('Analyse par Position', fontsize=16, fontweight='bold')

# Top 15 positions by frequency
ax1 = axes[0]
position_counts = df['Position_1986'].value_counts().head(15)
ax1.barh(range(len(position_counts)), position_counts.values, color='teal', edgecolor='black')
ax1.set_yticks(range(len(position_counts)))
ax1.set_yticklabels(position_counts.index)
ax1.set_xlabel('Number of Players')
ax1.set_title('Top 15 Positions les Plus Fréquentes')
ax1.grid(True, alpha=0.3, axis='x')

# Average salary by top positions
ax2 = axes[1]
top_positions = position_counts.head(10).index
salary_by_position = df[df['Position_1986'].isin(top_positions)].groupby('Position_1986')['Salary_1987'].mean().sort_values(ascending=False)
ax2.bar(range(len(salary_by_position)), salary_by_position.values, color='orange', edgecolor='black')
ax2.set_xticks(range(len(salary_by_position)))
ax2.set_xticklabels(salary_by_position.index, rotation=45)
ax2.set_ylabel('Average Salary_1987 ($ thousands)')
ax2.set_title('Salaire Moyen par Position (Top 10)')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '06_position_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '06_position_analysis.png'}")
plt.close()

# ---- Figure 7: Career vs 1986 Performance ----
fig7, axes = plt.subplots(2, 2, figsize=(14, 12))
fig7.suptitle('Comparaison Performance 1986 vs Carrière', fontsize=16, fontweight='bold')

comparisons = [
    ('Hits_86', 'Hits_career'),
    ('Home_runs_1986', 'Home_runs_career'),
    ('Runs_1986', 'Runs_career'),
    ('Runs_batted_1986', 'Runs_batted_career')
]

for idx, (var_86, var_career) in enumerate(comparisons):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    mask = df[var_86].notna() & df[var_career].notna()
    x = df.loc[mask, var_86]
    y = df.loc[mask, var_career]
    
    ax.scatter(x, y, alpha=0.5, s=30, color='darkblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line for reference
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5, label='x=y')
    
    ax.set_xlabel(f'{var_86}', fontsize=10)
    ax.set_ylabel(f'{var_career}', fontsize=10)
    ax.set_title(f'{var_86} vs {var_career}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(output_dir / '07_career_vs_1986.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '07_career_vs_1986.png'}")
plt.close()

# ---- Figure 8: Team Analysis ----
fig8, axes = plt.subplots(1, 2, figsize=(16, 6))
fig8.suptitle('Analyse par Équipe', fontsize=16, fontweight='bold')

# Number of players by team
ax1 = axes[0]
team_counts = df['Team_1986'].value_counts().head(15)
ax1.barh(range(len(team_counts)), team_counts.values, color='steelblue', edgecolor='black')
ax1.set_yticks(range(len(team_counts)))
ax1.set_yticklabels(team_counts.index)
ax1.set_xlabel('Number of Players')
ax1.set_title('Top 15 Équipes par Nombre de Joueurs')
ax1.grid(True, alpha=0.3, axis='x')

# Average salary by team (top 10)
ax2 = axes[1]
top_teams = team_counts.head(10).index
salary_by_team = df[df['Team_1986'].isin(top_teams)].groupby('Team_1986')['Salary_1987'].mean().sort_values(ascending=False)
ax2.barh(range(len(salary_by_team)), salary_by_team.values, color='coral', edgecolor='black')
ax2.set_yticks(range(len(salary_by_team)))
ax2.set_yticklabels(salary_by_team.index)
ax2.set_xlabel('Average Salary_1987 ($ thousands)')
ax2.set_title('Salaire Moyen par Équipe (Top 10)')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / '08_team_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '08_team_analysis.png'}")
plt.close()

# ============================================================================
# 9. SUMMARY STATISTICS EXPORT
# ============================================================================

print("\n" + "="*80)
print("9. EXPORTING SUMMARY STATISTICS")
print("="*80 + "\n")

# Create comprehensive summary
summary_stats = df[numeric_cols].describe().T
summary_stats['CV'] = (summary_stats['std'] / summary_stats['mean']) * 100
summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
summary_stats['Range'] = summary_stats['max'] - summary_stats['min']

summary_stats.to_csv(output_dir / 'summary_statistics.csv')
print(f"✓ Saved: {output_dir / 'summary_statistics.csv'}")

# Export correlation matrix
correlation_df.to_csv(output_dir / 'correlation_matrix.csv')
print(f"✓ Saved: {output_dir / 'correlation_matrix.csv'}")

# Export frequency tables for categorical variables
with open(output_dir / 'categorical_frequencies.txt', 'w') as f:
    for col in categorical_cols:
        f.write(f"\n{'='*80}\n")
        f.write(f"{col}\n")
        f.write(f"{'='*80}\n")
        freq = df[col].value_counts()
        freq_pct = (freq / len(df)) * 100
        freq_df = pd.DataFrame({
            'Count': freq,
            'Percentage': freq_pct
        })
        f.write(freq_df.to_string())
        f.write(f"\n\nTotal unique values: {df[col].nunique()}\n")

print(f"✓ Saved: {output_dir / 'categorical_frequencies.txt'}")

# ============================================================================
# 10. KEY FINDINGS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("10. KEY FINDINGS SUMMARY")
print("="*80 + "\n")

print("Dataset Overview:")
print(f"  - Total players: {len(df)}")
print(f"  - Players with salary data: {df['Salary_1987'].notna().sum()} ({df['Salary_1987'].notna().sum()/len(df)*100:.1f}%)")
print(f"  - Number of teams: {df['Team_1986'].nunique()}")
print(f"  - Number of positions: {df['Position_1986'].nunique()}")

print("\n\nSalary Statistics (1987):")
salary_stats = df['Salary_1987'].describe()
print(f"  - Mean: ${salary_stats['mean']:.2f}k")
print(f"  - Median: ${salary_stats['50%']:.2f}k")
print(f"  - Min: ${salary_stats['min']:.2f}k")
print(f"  - Max: ${salary_stats['max']:.2f}k")
print(f"  - Std Dev: ${salary_stats['std']:.2f}k")

print("\n\nTop Correlations with Salary:")
top_corr = correlation_df['Salary_1987'].sort_values(ascending=False)
for var, corr in list(top_corr.items())[1:6]:  # Skip Salary_1987 itself
    print(f"  - {var}: {corr:.3f}")

print("\n\nMost Common Positions:")
for pos, count in df['Position_1986'].value_counts().head(5).items():
    print(f"  - {pos}: {count} players ({count/len(df)*100:.1f}%)")

print("\n\nLeague Distribution:")
for league, count in df['League_1986'].value_counts().items():
    print(f"  - {league}: {count} players ({count/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("EDA COMPLETE - All visualizations and summaries saved to:")
print(f"{output_dir.absolute()}")
print("="*80)
print("\nGenerated files:")
print("  1. 01_distributions.png - Histograms of key variables")
print("  2. 02_boxplots.png - Box plots for outlier detection")
print("  3. 03_correlation_heatmap.png - Correlation matrix")
print("  4. 04_scatter_performance_salary.png - Performance vs Salary scatter plots")
print("  5. 05_league_division_analysis.png - League and division analysis")
print("  6. 06_position_analysis.png - Position frequency and salary analysis")
print("  7. 07_career_vs_1986.png - Career vs 1986 performance comparison")
print("  8. 08_team_analysis.png - Team-based analysis")
print("  9. summary_statistics.csv - Complete statistical summary")
print(" 10. correlation_matrix.csv - Correlation coefficients")
print(" 11. categorical_frequencies.txt - Frequency tables")
print("="*80)
print("\nTo view your images:")
print(f"1. Open your file explorer")
print(f"2. Navigate to: {output_dir.absolute()}")
print(f"3. Double-click any .png file to view it")
print("="*80)

# ============================================================================
# EXTRA — Salary distribution only (for presentation)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

salary_data = df['Salary_1987'].dropna()

ax.hist(salary_data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(salary_data.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Moyenne: {salary_data.mean():.1f}')
ax.axvline(salary_data.median(), color='green', linestyle='--', linewidth=2, 
           label=f'Médiane: {salary_data.median():.1f}')

ax.set_xlabel('Salary_1987 (milliers de dollars)', fontsize=11)
ax.set_ylabel('Fréquence', fontsize=11)
ax.set_title('Distribution des Salaires 1987', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'salary_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: salary_distribution.png")
plt.close()

# ============================================================================
# EXTRA — Salary boxplot only (for presentation)
# ============================================================================

fig, ax = plt.subplots(figsize=(6, 6))

salary_data = df['Salary_1987'].dropna()

bp = ax.boxplot(salary_data, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)

ax.set_ylabel('Salary_1987 (milliers de dollars)', fontsize=11)
ax.set_title('Box Plot : Salary_1987', fontsize=13, fontweight='bold')
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'salary_boxplot.png', dpi=150, bbox_inches='tight')
print("✓ Saved: salary_boxplot.png")
plt.close()