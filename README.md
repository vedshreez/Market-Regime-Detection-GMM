# Market Regime Detection via Gaussian Mixture Models (GMM)

**Status:** Completed (Dec 2025)
**Tech Stack:** Python (Scikit-Learn, PCA), Bloomberg Data, FRED API

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Key Results & Visuals](#2-key-results--visuals-the-alpha)
3. [Methodology](#3-methodology)
4. [Execution Roadmap](#4-execution-roadmap-how-to-run)
5. [References](#5-references)

---

## 1. Executive Summary
Financial markets exhibit distinct "regimes"—periods of calm (low correlation) vs. crisis (high correlation). Traditional models often fail to detect these shifts early.

In this project, we built an unsupervised learning framework to:
1.  **Reduce Dimensionality:** Applied PCA to 52 macro-financial factors.
2.  **Cluster Regimes:** Used Gaussian Mixture Models (GMM) to identify "Calm" vs. "Crisis" states.
3.  **Validate:** Proved distinct correlation structures with a **0.192 Silhouette Score**.

---

## 2. Key Results & Visuals (The "Alpha")
Our model successfully identified two distinct volatility regimes:
* **Regime 0 (Steady State):** Characterized by low volatility and negative stock-bond correlation. (~70% of history).
* **Regime 1 (Crisis State):** Characterized by high volatility clustering and correlation breakdowns. (~30% of history).

### Visual Analysis
*The chart below demonstrates the model's ability to switch regimes during the 2008 Crisis and 2020 COVID Crash.*

![Regime Transition Plot](gmm_plots/YOUR_EXACT_FILENAME.png)
*(Note: Replace 'YOUR_EXACT_FILENAME.png' with your actual file name)*

**Validation Metrics:**
* **Silhouette Score:** 0.192 (Indicates strong cluster separation).
* **BIC Score:** Minimized at K=2 regimes.
* **Performance:** Correctly flagged the Dot-Com Bubble (2000), 2008 Financial Crisis, and COVID-19 Crash.

---

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

---

## 4. Execution Roadmap (How to Run)
To replicate this analysis, follow this sequence:

### Step 1: Environment Setup 
```bash
pip install -r requirements.txt

---

### **Step 2**: Data Initialization
* **Database:** Run `python init_db.py` to set up the SQLite storage.
* **Ingestion:** Run `python load_fred.py` and `python load_bloomberg.py` to fetch raw indicators.

### Step 3: Feature Engineering
* Run `python compute_returns.py` to calculate log-returns and rolling volatility.
* Run `python pca.py` to compress the 52 factors into Principal Components.

### Step 4: Model Training (GMM)
Run the clustering algorithm:
```bash
python gmm.py --start 1995-01-01 --kmin 2 --kmax 6
