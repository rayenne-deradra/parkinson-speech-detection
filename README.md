# 🎙️ Parkinson's Disease Detection from Vocal Biomarkers

Complete machine learning pipeline for binary classification of Parkinson's disease using acoustic voice measurements — from raw data to clinical optimization.

---

## 📋 Dataset

**Source:** UCI Machine Learning Repository — Parkinson's Disease Voice Measurements (Max Little, University of Oxford, 2008)  
🔗 https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/

| Property | Value |
|----------|-------|
| Recordings | 195 |
| Subjects | 31 (23 with PD, 8 healthy) |
| Features | 22 acoustic measurements |
| Target | `status` — 1 = Parkinson's, 0 = Healthy |
| Class distribution | 147 PD (75.4%) / 48 Healthy (24.6%) |
| Missing values | 0 |
| Duplicates | 0 |

### Feature Groups

| Group | Features |
|-------|---------|
| Frequency | MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz) |
| Jitter (frequency perturbation) | MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP |
| Shimmer (amplitude perturbation) | MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3/5, MDVP:APQ, Shimmer:DDA |
| Noise measures | NHR, HNR |
| Nonlinear dynamical | RPDE, DFA, spread1, spread2, D2, PPE |

---

## 🔬 Pipeline Overview

```
Data Loading & Inspection
    → Cleaning (missing values · duplicates · Z-score + IQR outliers)
        → Descriptive Statistics + Spearman Correlation Heatmap
            → Bivariate Analysis (Shapiro-Wilk → Mann-Whitney U / t-test)
                → Feature Selection (Random Forest + LASSO + RFE + VIF)
                    → Classification (6 models · RandomizedSearchCV · 10-fold CV)
                        → AIC / BIC comparison
                            → SHAP Interpretation
                                → Six Sigma Pre/Post Analysis
                                    → Threshold Optimization
```

---

## 📊 Results

### Bivariate Analysis
All 22 features are statistically significant (p < 0.05). Mann-Whitney U test was used for nearly all features due to non-normality (Shapiro-Wilk). Key directions:
- **Jitter, Shimmer, NHR, RPDE, PPE, spread1** → higher in Parkinson patients
- **HNR, MDVP:Fo(Hz)** → lower in Parkinson patients

### Feature Selection
Three methods combined (RF importance + LASSO + RFE) selected **14 final features**:

| Method | Features selected | Notes |
|--------|------------------|-------|
| Random Forest (top 10) | PPE, spread1, Shimmer:APQ5, MDVP:APQ, spread2, MDVP:Shimmer, MDVP:RAP, Jitter:DDP, D2, MDVP:Fo(Hz) | Nonlinear features dominate |
| LASSO (L1, C=0.1) | spread1, spread2, D2, MDVP:Fo(Hz), MDVP:Flo(Hz) | Strict regularization |
| RFE (top 10) | MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, Jitter:DDP, MDVP:APQ, NHR, spread1, D2, PPE | Balanced selection |

**VIF before feature selection:** up to 15,000,000 (Shimmer:APQ3, Shimmer:DDA — near-perfectly correlated). Substantially reduced after selection.

### AIC / BIC (Logistic Regression)

| Feature Set | Num Features | AIC | BIC |
|-------------|-------------|-----|-----|
| **Feature-Selected** | **14** | **141.33 ✅** | **187.08 ✅** |
| All Features | 22 | 153.58 | 223.72 |

Feature-selected model wins on both criteria — more parsimonious with better generalization.

### Classification — Feature-Selected Set (10-Fold CV)

| Model | AUC | Sensitivity | Specificity | F1 | PR-AUC |
|-------|-----|-------------|-------------|----|--------|
| **SVM (RBF)** | **0.948 ± 0.062** | **0.958 ± 0.059** | 0.725 ± 0.233 | **0.938 ± 0.053** | **0.985** |
| Random Forest | 0.938 ± 0.083 | 0.933 ± 0.077 | 0.708 ± 0.212 | 0.921 ± 0.055 | 0.982 |
| XGBoost | 0.933 ± 0.085 | 0.924 ± 0.083 | **0.808 ± 0.239** | 0.931 ± 0.055 | 0.978 |
| AdaBoost | 0.910 ± 0.110 | 0.948 ± 0.044 | 0.675 ± 0.313 | 0.926 ± 0.049 | 0.973 |
| Logistic Regression | 0.902 ± 0.111 | 0.823 ± 0.114 | 0.833 ± 0.245 | 0.875 ± 0.080 | 0.970 |
| SVM (Linear) | 0.889 ± 0.114 | 0.840 ± 0.143 | 0.833 ± 0.245 | 0.884 ± 0.095 | 0.964 |

### Classification — Feature-Selected Set (Test Set, n=39)

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC | FN | FP |
|-------|----------|-------------|-------------|-----|-----|----|----|
| **Random Forest** | **0.923** | **0.966** | **0.800** | **0.949** | **0.962** | 1 | 2 |
| SVM (RBF) | 0.872 | 0.897 | 0.800 | 0.912 | 0.938 | 3 | 2 |
| AdaBoost | 0.795 | 0.862 | 0.600 | 0.862 | 0.945 | 4 | 4 |

### Classification — All Features (10-Fold CV)

| Model | AUC | Sensitivity | Specificity | F1 |
|-------|-----|-------------|-------------|----|
| **XGBoost** | **0.970 ± 0.051** | 0.949 ± 0.059 | **0.842 ± 0.182** | **0.949 ± 0.034** |
| AdaBoost | 0.965 ± 0.054 | **0.958 ± 0.059** | 0.783 ± 0.258 | 0.946 ± 0.038 |
| SVM (RBF) | 0.960 ± 0.051 | 0.992 ± 0.026 | 0.317 ± 0.156 ⚠️ | 0.897 ± 0.025 |

> ⚠️ SVM (RBF) on all features achieves very high Sensitivity (0.992) but very low Specificity (0.317) — it nearly always predicts Parkinson, a known behavior under class imbalance with high-dimensional correlated features. This confirms the value of feature selection.

### SHAP Interpretation (SVM RBF — Feature-Selected)

Most impactful features for predicting Parkinson:
- **spread1, PPE** — highest positive impact (high values → PD prediction)
- **RPDE, D2** — nonlinear dynamical complexity markers, elevated in PD
- **HNR** — negative impact (high HNR = healthy voice → pushes against PD prediction)
- **DFA** — threshold-like effect, elevated in PD

### Six Sigma Analysis (Pre/Post — Feature Stabilization Simulation)

Simulates a 20% shift of test features toward healthy-class means (α=0.20):

| Model | Sigma PRE | Sigma POST | Improvement | McNemar p |
|-------|-----------|------------|-------------|-----------|
| Random Forest | 2.926 | 2.635 | -0.291 | 0.500 |
| AdaBoost | 2.323 | 2.635 | **+0.311** | 0.375 |
| SVM (RBF) | 2.635 | 2.520 | -0.115 | 1.000 |

No statistically significant differences found (McNemar p > 0.05). This is expected given the small test set (n=39) — a larger longitudinal dataset would be required to detect meaningful signal from vocal stabilization.

### Threshold Optimization (Sensitivity ≥ 0.95 constraint)

| Model | Baseline τ | Optimal τ | Sensitivity | Sigma | Improvement |
|-------|-----------|-----------|-------------|-------|-------------|
| Random Forest | 0.50 | 0.50 | 0.966 | 2.926 | already optimal |
| AdaBoost | 0.50 | **0.35** | **0.966** | 2.635 | +0.311 sigma |
| SVM (RBF) | 0.50 | — | < 0.95 at all thresholds | — | — |

Random Forest at default threshold already achieves Sensitivity=0.966 — only 1 missed PD patient out of 29 in the test set.

---

## 🛠️ How to Run

### Google Colab (recommended)
```python
# The notebook downloads the dataset automatically:
import urllib.request
urllib.request.urlretrieve(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
    'parkinsons.csv'
)
```
Then run all cells. Expected runtime: ~60–90 min (CPU) or ~30–40 min with N_ITER=20.

### Local
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap statsmodels scipy joblib
jupyter notebook parkinson_complete.ipynb
```

---

## 📦 Dependencies

```
pandas · numpy · matplotlib · seaborn
scikit-learn · xgboost · shap
scipy · statsmodels · joblib
```

---

## 🔑 Key Takeaways

- **PPE and spread1** are the single most discriminative vocal biomarkers across all methods
- **Feature selection is essential** — reduces VIF from 15,000,000 to manageable levels and improves model interpretability
- **Random Forest** achieves the best test-set balance: Sensitivity=0.966, Specificity=0.800, only 1 missed PD patient
- **Threshold optimization** at τ=0.35 brings AdaBoost to Sensitivity=0.966 — useful in high-recall screening scenarios
- **Six Sigma analysis** confirms that with n=39 test samples, statistical significance of improvements cannot be established — longitudinal data with larger cohorts is needed

---

## 🚀 Relevance to Longitudinal Research (EvoluPark)

This project establishes a robust baseline for vocal biomarker analysis in Parkinson's disease. Planned extensions toward a full longitudinal pipeline:

1. **Temporal modelling** — LSTM / Transformer models on repeated recordings (e.g. every 6 months)
2. **Pretrained speech models** — Wav2Vec 2.0 / Whisper embeddings as richer feature extractors
3. **Paraverbal features** — speech rate, pause duration, disfluencies
4. **Clinical correlation** — link vocal markers to UPDRS and MoCA scores
5. **Inter-patient generalization** — meta-learning / few-shot approaches for new patients with limited data

---

## 👩‍💻 Author

**Rayenne Maissoune Deradra**  
MSc Computer Science — Data Science, Université Ferhat Abbas Sétif-1, Algeria  
rayennederadra@gmail.com

---

## 📄 References

Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007).
*Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection.*
BioMedical Engineering OnLine, 6:23.
