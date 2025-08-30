# Machine Learning Projects

## Overview
This repository contains multiple **Machine Learning** projects covering classification, clustering, dimensionality reduction, and regression tasks. Each notebook follows an **end-to-end** machine learning pipeline, including **data preprocessing, feature engineering, model training, evaluation, and predictions**.

## Included Notebooks

1. **Titanic Survival Prediction (`Titanic_train_survival_predictions.ipynb`)**
   - Predicts passenger survival using classification models based on passenger details.

2. **Loan Amount Analysis (`LoanAmount_Analysis.ipynb`)**
   - Analyzes and predicts loan eligibility using regression and classification techniques.

3. **K-Means Clustering (`K-means Clusturing (1).ipynb`)**
   - Performs customer segmentation using the K-Means clustering algorithm.

4. **Principal Component Analysis (PCA) (`PCA@digits.ipynb`)**
   - Reduces dimensionality of the **Digits dataset** while preserving key patterns.

5. **Support Vector Machine (SVM) (`SVM.ipynb`)**
   - Implements an SVM classifier for a given dataset to perform supervised learning.

## Dependencies
Ensure you have the following libraries installed before running the notebooks:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Preprocessing Steps
Each project follows a structured **data preprocessing pipeline**, including:
- **Data Loading**: Import datasets using Pandas or Scikit-learn utilities.
- **Handling Missing Values**: Fill or drop missing values based on analysis.
- **Feature Engineering**: Extract and transform important features.
- **Scaling & Encoding**:
  - Standardization or normalization for numerical features.
  - Encoding categorical variables using One-Hot Encoding or Label Encoding.
- **Splitting Data**: Dividing data into training and test sets.

## Model Training & Validation
Each project includes:
- Training various **supervised and unsupervised learning models**.
- Performing **hyperparameter tuning** to improve accuracy.
- Evaluating models using:
  - **Classification Metrics**: Accuracy, Precision, Recall, F1-score.
  - **Regression Metrics**: RMSE, MAE, R-squared.
  - **Clustering Metrics**: Inertia, Silhouette Score.

## Expected Results
- **Titanic Survival Prediction**: Classification of passengers into survivors/non-survivors.
- **Loan Amount Analysis**: Predict whether an applicant is eligible for a loan.
- **K-Means Clustering**: Segmentation of data points into meaningful clusters.
- **PCA on Digits**: Visualization of high-dimensional data in reduced form.
- **SVM Model**: Predictive classification using Support Vector Machines.

## How to Use
- Clone this repository and navigate to the respective notebook.
- Run the Jupyter notebook step by step to execute preprocessing, training, and evaluation.
- Modify parameters and features to experiment with different results.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn for Visualization](https://seaborn.pydata.org/)

