# QoS-Privacy Paradox: How Modern Web Optimization Shapes Website Fingerprinting

This repository contains the official implementation and experimental code for our research paper investigating the relationship between Quality of Service (QoS) metrics and website fingerprinting (WF) vulnerability.

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Experimental Pipeline](#experimental-pipeline)
- [Dependencies](#dependencies)
- [Data Description](#data-description)
- [Analysis Modules](#analysis-modules)

## 🔍 Overview

This research explores the fundamental tension between web performance optimization and privacy leakage through website fingerprinting attacks. We investigate how modern QoS improvements paradoxically increase websites' vulnerability to traffic analysis attacks.

### Key Contributions
- **Comprehensive Dataset**: Collection of QoS metrics from 1,500 popular websites
- **Multi-model Evaluation**: Implementation and comparison of 5 state-of-the-art WF attack models
- **Causal Analysis**: Propensity Score Matching to establish causal relationships
- **Feature Attribution**: SHAP analysis to understand leakage mechanisms

## 📁 Repository Structure

```
qos_wf/
├── data_preprocess/          # Data preprocessing and extraction scripts
├── wf_experiment/           # Website fingerprinting attack implementations
│   ├── RF/                  # Random Forest-based attack
│   ├── DF/                  # Deep Fingerprinting attack
│   ├── STAR/                # STAR attack implementation
│   ├── FineWP/              # Fine-grained Web Page attack
│   └── kfp/                 # k-FP attack implementation
├── correlation/             # Statistical correlation analysis
├── SHAP/                    # SHAP value computation and visualization
├── VIF/                     # Variance Inflation Factor analysis
├── PSM/                     # Propensity Score Matching for causal inference
└── README.md
```


## ⚙️ Experimental Pipeline

The research follows a systematic pipeline:

1. **Traffic Collection**: Capture network traces from 1,500 websites
2. **QoS Measurement**: Extract performance metrics (Load Time, TTFB, etc.)
3. **Attack Training**: Train 5 different WF models on collected traces
4. **Vulnerability Assessment**: Evaluate F1 scores across all websites
5. **Statistical Analysis**: Correlation, causality, and attribution analysis

## 📦 Dependencies

### Core Requirements
```bash
Python >= 3.8
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
xgboost
shap
```

### Deep Learning Frameworks
```bash
torch >= 1.8
tensorflow >= 2.0  # for DF model
```

### Network Analysis
```bash
scapy
```

Install all dependencies:
```bash
pip install -r requirements.txt
```


## 📊 Data Description

### Input Data
- **Website List**: 1,500 popular websites from Tranco rankings
- **Network Traces**: Captured using tcpdump/Wireshark
- **Performance Metrics**: Load time, TTFB, DNS lookup, TLS handshake, etc.
- **Runtime Metrics**: JavaScript execution time, paint count, layout operations

### Output Data
- **F1 Scores**: Per-website vulnerability scores for each attack model
- **Feature Importance**: SHAP values for key metrics
- **Causal Effects**: PSM-derived treatment effects
- **Correlation Matrices**: Spearman/Pearson correlation coefficients

### Key Metrics Tracked
- **Web QoS**: HTTP3 adoption, CDN usage, connection reuse
- **Performance**: Load time, TTFB, DNS resolution time
- **Resource**: Total bytes transferred, request count
- **Runtime**: JS execution time, paint/layout operations
- **QoE**: LCP, FCP, CLS, Speed Index

## 🔬 Analysis Modules

### 1. Correlation Analysis (`correlation/`)
- Spearman/Pearson correlation between QoS metrics and WF vulnerability
- Mutual information analysis for non-linear relationships
- Per-attack model correlation patterns

### 2. Causal Inference (`PSM/`)
- Propensity Score Matching to establish causality
- Average Treatment Effect estimation
- Confounding variable control

### 3. Feature Attribution (`SHAP/`)
- SHAP value computation for model interpretability
- Hierarchical R² analysis
- Interaction effect visualization

### 4. Multicollinearity Analysis (`VIF/`)
- Variance Inflation Factor calculation
- Partial correlation analysis
- Feature independence assessment

## 📈 Visualization Examples

The repository includes various visualization scripts:
- Heatmaps for correlation matrices
- Bar plots for feature importance
- Line charts for trend analysis
- Scatter plots with confidence intervals

## 🔧 Customization

### Adding New Attack Models
1. Create new directory in `wf_experiment/`
2. Implement training and evaluation scripts
3. Save results in standardized `.npz` format
4. Update analysis scripts to include new model

### Extending QoS Metrics
1. Modify data preprocessing scripts
2. Update feature extraction pipelines
3. Adjust correlation analysis modules



---

*This repository represents work conducted as part of academic research. All findings are presented for educational and scientific purposes.*
