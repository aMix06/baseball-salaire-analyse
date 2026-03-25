# Baseball Salary Analysis

A data analysis project completed as part of the **ARE Biostat** course at Sorbonne University.

## Research Question

> **Are baseball players paid according to their performance?**

Using a dataset of Major League Baseball players from 1986–1987, we investigate whether player performance statistics can explain salary differences.

## Dataset

- **Source:** Sports Illustrated, April 20, 1987 & The 1987 Baseball Encyclopedia Update (Collier Books / Macmillan)
- **Reference paper:** Hoaglin & Velleman (1995) — *"A Critical Look at Some Analyses of Major League Baseball Salaries"*
- **Size:** 322 players (batters who regularly played in 1986)
- **Key variables:** Season stats (hits, home runs, RBIs, walks), career stats, longevity, and 1987 salary

## Methods

### Variable Transformations
- `log(Salary)` — applied to correct right skew in the salary distribution
- `Hits_per_year = Hits_career / Longevity` — normalized career hits by years played to capture consistency

### Regression Models
We built four models to progressively understand what drives salary:

| Model | Variables |
|-------|-----------|
| M1 | log(Salary) ~ Longevity |
| M2 | log(Salary) ~ Hits_86 |
| M3 | log(Salary) ~ Hits_per_year |
| M4 | log(Salary) ~ Longevity + Hits_per_year + Hits_86 |

Models M1–M3 isolate individual predictors. M4 combines the strongest predictors into a multiple regression model.

### Diagnostics
Each model is evaluated using R², p-values, and residual plots.

## Project Structure

```
baseball-salary-analysis/
│
├── Baseball.csv                    # Raw dataset
├── baseball_eda_local.py           # Exploratory data analysis (distributions, correlations, visualizations)
├── baseball_regression_final.py    # Final regression models and diagnostics
│
└── eda_outputs/                    # Generated plots and summary statistics (auto-created on run)
```

## How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scipy
```

**2. Run EDA:**
```bash
python baseball_eda_local.py
```

**3. Run regression analysis:**
```bash
python baseball_regression_final.py
```

Output plots and statistics will be saved to the `eda_outputs/` folder.

## Authors

Project completed in a group of 3 as part of the ARE Biostat course, Sorbonne University (2025–2026).
