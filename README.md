# Market Regime Detection via Gaussian Mixture Models (GMM)

**Status:** Completed (Dec 2025)
**Tech Stack:** Python (Scikit-Learn, PCA), Bloomberg Data, FRED API

## 1. Executive Summary
Financial markets exhibit distinct "regimes"—periods of calm (low correlation) vs. crisis (high correlation). Traditional models often fail to detect these shifts early.

In this project, we built an unsupervised learning framework to:
1.  **Reduce Dimensionality:** Applied PCA to 52 macro-financial factors.
2.  **Cluster Regimes:** Used Gaussian Mixture Models (GMM) to identify "Calm" vs. "Crisis" states.
3.  **Validate:** Proved distinct correlation structures with a **0.192 Silhouette Score**.

## 2. Key Results (The "Alpha")
Our model successfully identified two distinct volatility regimes:
* **Regime 0 (Steady State):** Characterized by low volatility and negative stock-bond correlation. (~70% of history).
* **Regime 1 (Crisis State):** Characterized by high volatility clustering and correlation breakdowns. (~30% of history).

**Validation Metrics:**
* **Silhouette Score:** 0.192 (Indicates strong cluster separation).
* **BIC Score:** Minimized at K=2 regimes.
* **Performance:** Correctly flagged the Dot-Com Bubble (2000), 2008 Financial Crisis, and COVID-19 Crash.

## 3. Methodology
### A. Data Engineering
* Aggregated data from **Bloomberg** (Equities, Volatility) and **FRED** (Macro indicators).
* Engineered rolling volatility features and "Trend Following" signals.

### B. Dimensionality Reduction (PCA)
* Applied Principal Component Analysis (PCA) to extract the dominant signal from 18 factor categories.
* Retained PC1 (First Principal Component) to reduce noise and prevent overfitting.

### C. GMM Clustering
* Unlike K-Means (which assumes spherical clusters), GMM allows for **elliptical clusters**, crucial for financial data where correlation creates skew.
* Used **Expectation-Maximization (EM)** to fit the model parameters.

## 4. Repository Structure
* `gmm.py`: Core logic for Gaussian Mixture Model training.
* `pca.py`: Dimensionality reduction pipeline.
* `regime_backtest.py`: Validating the regimes against historical market crashes.
* `visualizations.py`: Generating the regime transition plots.

## 5. Credits
* **Code & Research:** Developed in collaboration with Isaiah Nick.
* **Data Sources:** Bloomberg, Kenneth French Library, St. Louis Fed.
