# Iris Classification Models

## Overview
This project applies multiple classification models to the Iris dataset and compares their performance using:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion matrices

It demonstrates:
- Loading datasets directly from `sklearn.datasets`
- Training and evaluating models: KNN, SVM, Logistic Regression, Decision Tree
- Displaying confusion matrices visually
- Creating a comparison table of metrics
- Short analysis of model performance

---

## Dataset
- 150 samples, 4 features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- 3 classes: Setosa, Versicolor, Virginica
- Loaded directly from `sklearn.datasets.load_iris()`, no CSV required.

---

## How It Works
1. Load the Iris dataset  
2. Explore dataset: shape, number of classes, class distribution  
3. Split dataset into training and testing sets (stratified)  
4. Train 4 models:
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Logistic Regression
   - Decision Tree
5. Evaluate models:
   - Accuracy
   - Classification report (precision, recall, F1-score)
   - Confusion matrix (displayed visually)  
6. Generate comparison table with macro metrics and accuracy  
7. Analysis of results

---
