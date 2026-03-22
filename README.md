#  Speech Recognition with HMMs & RNNs

## Overview
This project implements a complete speech recognition pipeline for isolated digit classification using both classical probabilistic models and deep learning approaches.

The system processes raw audio signals, extracts meaningful acoustic features, and performs classification using multiple machine learning techniques, including Hidden Markov Models (HMMs) and Recurrent Neural Networks (RNNs).

---

##  Key Features
- Audio preprocessing and feature extraction from raw `.wav` files
- MFCC, delta, and delta-delta feature engineering
- Dimensionality reduction with PCA
- Multiple classification approaches:
  - Bayesian classifiers
  - Naive Bayes (scikit-learn)
  - Custom ML models
- Sequence modelling using:
  - GMM-HMM (pomegranate)
  - RNN / LSTM (PyTorch)
- Model evaluation:
  - Accuracy metrics
  - Confusion matrices
  - Visualization of feature spaces

---

##  Methodology

### 1. Audio Processing
- Input: speech recordings of spoken digits
- Sampling frequency: 16 kHz
- Feature extraction using `librosa`

### 2. Feature Engineering
- Mel-Frequency Cepstral Coefficients (MFCCs)
- First and second temporal derivatives (Δ, ΔΔ)
- Statistical aggregation (mean & variance per utterance)

### 3. Dimensionality Reduction
- Principal Component Analysis (PCA)
- 2D and 3D visualization of feature space

### 4. Classical Machine Learning
- Bayesian classifier
- Naive Bayes
- Additional classifiers (e.g. SVM / k-NN depending on implementation)

### 5. Sequence Modelling

#### 🔹 GMM-HMM
- Left-to-right Hidden Markov Models
- Gaussian Mixture emissions
- Trained via Expectation-Maximization (EM)

#### 🔹 Recurrent Neural Networks
- LSTM-based sequence prediction
- Regularization techniques:
  - Dropout
  - L2 regularization
  - Early stopping
- Bidirectional LSTM for improved temporal context

---

##  Results
- Comparison of multiple classifiers on test data
- Confusion matrices for performance evaluation
- Analysis of feature separability and model performance

---

##  Tech Stack
- Python
- NumPy, SciPy
- librosa (audio processing)
- scikit-learn (ML models)
- PyTorch (deep learning)
- matplotlib / seaborn (visualization)
- pomegranate (HMMs)

---

## 📁 Project Structure
