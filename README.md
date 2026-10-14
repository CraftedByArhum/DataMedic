# DataMedic

**Automated Machine Learning Data Diagnostics and Human-in-the-Loop Benchmark Platform**

DataMedic is a data-centric machine learning diagnostic tool built with Python, Scikit-Learn, and Streamlit. Rather than relying on static rule-based filters, DataMedic applies unsupervised and out-of-fold supervised models to identify structural quality issues in tabular classification datasets. It pairs statistical anomaly attribution with a strict held-out evaluation pipeline to measure whether repairing flagged data improves downstream model generalization.

## Problem Statement

Machine learning pipelines frequently underperform due to underlying training data defects such as mislabeled records, multivariate outliers, duplicate entries, and target leakage. Standard exploratory data analysis checks columns in isolation and fails to measure how data cleaning decisions affect unseen test performance. 

DataMedic addresses this by locking away a 25% stratified test split immediately upon dataset upload. All diagnostic algorithms and user-approved repairs operate exclusively on the 75% training split, ensuring zero data leakage during evaluation.

## Core Machine Learning Capabilities

1. **Target Leakage Screening**
   Computes feature-to-target correlation and variance metrics prior to row-level checks. Features that strongly encode target labels are flagged for removal so they do not mask label noise during cross-validation.

2. **Label Error Detection (Confident Learning Principle)**
   Executes 5-fold stratified cross-validation using an out-of-fold Random Forest ensemble. Because a record is never evaluated by a model trained on that same record, rows where out-of-sample predicted probabilities strongly contradict the assigned label are flagged as suspected label errors.

3. **Multivariate Outlier Detection**
   Fits an Isolation Forest across standardized feature space to isolate multi-dimensional anomalies. For each flagged record, the engine calculates feature-level Z-scores to identify the primary variable driving the anomaly.

4. **Near-Duplicate Identification**
   Uses K-Nearest Neighbors with Euclidean distance over normalized numeric and encoded categorical features to detect exact duplicates and near-duplicate records caused by data entry jitter.

## Key Features

* **Universal CSV Preprocessing:** Automatically handles numeric imputation, categorical encoding, high-cardinality ID exclusion, and continuous target binning.
* **Evidence-Based Explanations:** Generates deterministic, plain-English explanations for every flagged row without requiring external LLM APIs or GPU hardware.
* **Human-in-the-Loop Review:** Provides interactive data tables where users can inspect individual records and approve or reject specific repairs row by row.
* **Experimental Validation:** Trains baseline and post-repair Random Forest models under identical conditions and compares Test Accuracy, Weighted F1-Score, Confusion Matrices, and Feature Distribution shifts.
* **Audit Lineage Export:** Exports the cleaned training dataset as a CSV alongside a timestamped Markdown audit report detailing every modified row index.

## Project Structure

* `ml_engine.py`: Core diagnostic classes, automated feature preprocessing, cross-validation probability estimation, outlier attribution, and duplicate search.
* `app.py`: Streamlit user interface, stratified train/test partitioning, interactive review tables, visual diagnostics, and report generation.
* `requirements.txt`: Python package dependencies required for local execution and cloud deployment.

## Local Setup and Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/CraftedByArhum/DataMedic.git](https://github.com/CraftedByArhum/DataMedic.git)
   cd DataMedic
