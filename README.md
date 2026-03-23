# Iris Classification Models

## Overview
This project applies multiple machine learning classification models to the Iris dataset and compares their performance.

The models used:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree

---

## Dataset
- 150 samples, 4 features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width  
- 3 classes:
  - Setosa
  - Versicolor
  - Virginica  
- Loaded directly using `sklearn.datasets.load_iris()`

---

## How It Works
1. Load and explore dataset  
2. Split data into training and testing sets  
3. Train multiple classification models  
4. Evaluate models using:
   - Accuracy  
   - Precision, Recall, F1-score  
   - Confusion Matrix  
5. Compare model performance  

---

## Results

- **KNN achieved 100% accuracy**
- **SVM, Logistic Regression, and Decision Tree achieved ~96.7% accuracy**

### Key Insights
- Setosa is perfectly classified in all models  
- Minor confusion occurs between Versicolor and Virginica  
- KNN performed the best on this dataset  

---

## Confusion Matrices

### KNN
![KNN](images/KNN_cm.png)

### SVM
![SVM](images/SVM_cm.png)

### Logistic Regression
![Logistic](images/Logistic_Regression_cm.png)

### Decision Tree
![Tree](images/Decision_Tree_cm.png)

---
