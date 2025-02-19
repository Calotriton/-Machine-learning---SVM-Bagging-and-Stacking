# Machine Learning Models: SVM, Bagging & Stacking

## Overview
A company specializing in used car sales faces the challenge of determining the optimal color for repainting vehicles arriving in poor condition. After evaluating options, they decide to limit choices to white and black, as these are the most common colors in the market. To automate this decision, the company plans to develop a predictive model that, based on the characteristics of second-hand vehicles, determines whether they were originally white or black.
This repository contains a machine learning project implementing **Support Vector Machines (SVM)**, **Bagging**, and **Stacking** for classification tasks. The dataset used includes various numerical and categorical features related to vehicle specifications, and the goal is to classify car colors.

## Models Implemented

### 1. **SVM with Grid Search Optimization**
- Applied **GridSearchCV** to optimize hyperparameters for **linear** and **RBF kernels**.
- Evaluated **accuracy** and **AUC (Area Under Curve)** on the test set.
- Plotted **Confusion Matrix** and **ROC Curve** for performance visualization.

### 2. **Bagging with SVM**
- Used **BaggingClassifier** to improve the robustness of the best SVM model.
- Compared performance against the standalone SVM model.

### 3. **Stacking Ensemble Model**
- Combined **Random Forest, Logistic Regression, and KNN** as base learners.
- Used **SVC (Support Vector Classifier)** as the meta-model.
- Evaluated **accuracy** and performed **cross-validation** to measure stability.

## Installation & Dependencies
To run the project, install the required Python libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-svm-bagging-stacking.git
   cd ml-svm-bagging-stacking
   ```
2. Run the main script:
   ```bash
   python main.py
   ```
3. View the model outputs, including accuracy metrics, confusion matrices, and plots.

## Results Summary
| Model       | Accuracy | AUC  |
|------------|---------|------|
| SVM (Best) | 0.6390  | 0.6392 |
| Bagging SVM | 0.6425  | 0.6426 |
| Stacking   | 0.6425  | - |

- **Bagging** slightly improved SVM’s performance.
- **Stacking** did not outperform the best individual model (Random Forest).
- **Random Forest performed best among base learners** in stacking.

## License
This project is open-source under the MIT License.

## Author
[Àlex López](https://github.com/your-username)


