#  Credit Card Fraud Detection: A Big Data Autoencoder Approach

**Course:** Advanced Machine Learning Techniques / University of Thessaly

An advanced, semi-supervised Machine Learning pipeline designed to detect fraudulent financial transactions. This project processes over 6.3 million records using a memory-efficient **Stateful Stream Processing** architecture and a **Deep Autoencoder** for anomaly detection.

---

## Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [System Architecture](#-system-architecture)
- [Model Evaluation & Results](#-model-evaluation--results)
- [How to Run](#-how-to-run)

---

## Overview
Fraud detection in banking systems typically suffers from extreme class imbalance and massive data volumes. Loading millions of records into RAM is often unfeasible for standard machines. 

This project tackles both issues by:
1. **Out-of-core Learning (Streaming):** Using Python Data Generators to process the 6.3M dataset in chunks, keeping RAM usage strictly under 500MB.
2. **Semi-supervised Anomaly Detection:** Training a Deep Autoencoder exclusively on *normal* transactions to learn the latent representations of legitimate behavior, flagging anything that produces a high Reconstruction Error (MSE) as fraud.

---

## 📊 Dataset
* **Source:** [PaySim - Synthetic Financial Datasets For Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)
* **Size:** 6,354,407 normal transactions and 8,213 fraudulent transactions.
* **Characteristics:** Highly imbalanced dataset (frauds account for ~0.13% of the total data).

---

## Feature Engineering
To empower the Autoencoder, standard anonymized features were not enough. We engineered domain-specific features based on real-world banking logic:

* **`hour_of_day`:** Extracted from the simulation `step` (1 step = 1 hour) to capture the time context of transactions (e.g., late-night transfers).
* **`transaction_freq`:** A stateful dictionary tracks the historical frequency of the `nameOrig` (sender). Fraudsters often exhibit high transaction velocity (testing cards, rapid cash-outs).
* **Mathematical Anomalies (`errorBalanceOrg`, `errorBalanceDest`):** Fraudulent "smash and grab" tactics often leave inconsistencies in the ledger. We explicitly calculated the difference between expected and actual balances for both sender and receiver.

---

## System Architecture
The pipeline is designed with a **Two-Pass Data Engineering** approach:
1. **Pass 1 (State Building):** Rapid scan of the dataset to build the `transaction_freq` dictionary.
2. **Pass 2 (Training via Generator):** Yielding data in chunks of 50,000 rows, applying transformations (`StandardScaler`, `OneHotEncoder`), and injecting the stateful frequency logic on the fly.

### Deep Autoencoder Configuration
An hourglass architecture was utilized to force maximum compression in the latent space:
* **Input Layer:** 14 Dimensions
* **Encoder:** Dense(32) -> Dense(16) -> Dense(6) *(Strict Bottleneck)*
* **Decoder:** Dense(16) -> Dense(32)
* **Output:** Dense(14)
* **Loss Function:** Mean Squared Error (MSE)
* **Optimization:** Adam with Early Stopping (patience=3)

---

## Model Evaluation & Results
Anomaly classification was determined by setting a strict threshold at the **99th percentile** of the Reconstruction Error (MSE) on normal transactions. This statistical boundary was chosen to intentionally limit False Positives to exactly 1%.

Evaluated on a massive Test Set of **508,213 records** (500,000 normal + 8,213 fraud):

| Metric | Score | Practical Meaning |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **55%** | Successfully identified **4,512** out of 8,213 total frauds. |
| **Precision** | **47%** | Nearly 1 in 2 alarms raised by the system is an actual fraud. |
| **False Positives** | **1.0%** | Exactly 5,000 false alarms out of 500,000 normal transactions, perfectly validating the 99th percentile threshold. |
| **F1-Score** | **0.51** | A highly robust balance for an extremely imbalanced Big Data problem. |

---

## How to Run

### 1. Prerequisites
Make sure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow
