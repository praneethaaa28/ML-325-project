# Student Performance Prediction Using Machine Learning


## Project Overview

The goal of this project is to predict whether a student will pass or fail (G3 >= 10) using study habits, attendance, and demographic features

The scope consists of preprocessing, feature engineering, binary classification, and interpretability via SHAP

## Technology Used
	
- Github
- VSCode
- Python
- NumPy
- Scikit-learn 
- Matplotlib / Seaborn

## Setup

Install Scikit-Learn, Shap, Randomforest, Matplotlib, Pandas, Seaborn, and XGBoost into the python runtime environment

## Dataset

- Overview
 - Total instances: 649
 - Features: 30
 - Missing values: 0
 - Target variable: G3 (final grade from 0–20)
- Key Dataset Notes
 - Two Portuguese secondary schools - Math & Portuguese subjects
 - G3 strongly correlated with G1 & G2
 - G1 & G2 excluded to prevent data leakage
 - Pass threshold: G3 >= 10 | Fail: G3 < 10
- Feature Categories
 - Demographic:	Sex, age, address, family size, cohabitation
 - Parental: Mother’s & Father’s education & occupations
 - School-Related: School choice reason, guardian, travel time
 - Behavioral/Lifestyle:  Study time, absences, alcohol use, internet, activities

## Team Contributions

- Kaitlin Sloan — Logistic Regression
  Build Logistic Regression pipeline
  Preprocess data (encoding + scaling)
  Train with/without G1 & G2
  Evaluate using F1, ROC-AUC, confusion matrix
  Write results summary + push notebook
  
- Praneetha Voona — Random Forest
  Build Random Forest pipeline
  Perform hyperparameter tuning + 5-fold CV
  Evaluate using F1, ROC-AUC, confusion matrix
  Plot feature importances
  Write results summary + push notebook
  
- Joshua Grewe — Gradient Boosting
  Build XGBoost/LightGBM pipeline
  Train using stratified 5-fold CV
  Compare with/without G1 & G2
  Evaluate using F1, ROC-AUC, confusion matrix
  Write results summary + push notebook
  
- Justin Gurkin — SHAP Fairness Analysis
  Train baseline Random Forest for SHAP
  Run SHAP analysis on features
  Analyze parental education bias impact
  Write results summary + push notebook
  
- Manihams Suraparaju — Ensemble + Report
  Build VotingClassifier ensemble (LR + RF + XGBoost)
  Evaluate using F1, ROC-AUC, confusion matrix
  Compile IEEE final report
  Push final notebook/report

## Preprocessing

- Data Quality Check:
 - Verified zero missing values across all 33 features
 - Ensured dataset was ready for immediate Modeling
- Creating Target Variable
 - Used G3 as Binary Classification: Pass = G3 >= 10, Fail = G3 <10
 - Dropped G1, G2, G3 from features to prevent data leakage
- Preprocessing Pipeline:
 - Categorical Features: OneHotEncoder(drop first category to avoid multicollinearity) 
 - Numeric Features: StandardScaler (mean =0, standard deviation = 1)
- Had an 80/20 train/test split with stratification to maintain class balance( random_state = 42)

## Logistic Regression

- Model Set Up: 
 - Established baseline performance for binary classification
 - Linear Model predicting probability of pass/ fail outcomes
 - Max iterations set to 500 to ensure convergence on preprocessed data
- Performance Metrics:
 - F1 Score 0.9346 (93.46%), meaning there was excellent precision-recall balance
 - ROC-AUC: 0.9790 (97.90%), meaning there was a near-perfect class separation 
- Confusion Matrix: 
- True Negatives: 22 |  False Positives: 4
- False Negatives: 3 |  True Positives: 50
- Only 7 total Misclassifications = 91.1 Accuracy
- Key Insight: Emphasizes that simple linear Models achieve strong performance 

## Random Forest

- Converted final grade (G3) into pass/fail label; removed G1, G2, and G3 to prevent data leakage.
- Built preprocessing pipeline: One-Hot Encoding (categorical) + Standardization (numeric).
- Performed 80/20 stratified train-test split to preserve class balance.
- Tuned the model by testing different parameter combinations using Grid Search with 5-fold cross-validation to find the best-performing setup.
- Evaluating using F1-score, ROC-AUC, and confusion matrix.
- Achieved improved classification performance over baseline.
- Key insight: Feature importance shows the strongest predictors of student success.

## Gradient Boosting

- XGBoost Model with stratified 5-fold cross-validation
- Two feature sets: with and without G1/G2
- Goal: measure impact of previous grades on predictions
- Compare realistic features vs prediction power
- Without G1/G2:
 - F1: 0.784, ROC-AUC: 0.625
 - Bias toward predicting “pass”
 - High pass recall (0.89), low fail recall (0.15)
- With G1/G2:
 - F1: 0.920, ROC-AUC: 0.942
 - Improved accuracy and class separation
 - Model was much more reliable
- G1 & G2 are strong features for predicting
- GB Model relies on previous performance, not behavior
- Dependence on G1/G2 is limiting since it may not always be available

## Model Evaluation Metrics

- F1 Score
- ROC-AUC
- Confusion Matrix
- Cross Validation Performance
- Feature Importance

## Findings

MODEL                     F1 SCORE    ROC-AUC     NOTES
Logistic Regression       0.9346      0.9790      Leaky baseline (kept G3)
Random Forest             0.7705      0.6067      No G1/G2/G3
XGBoost (without G1/G2)   0.7840      0.6248      Leak-free,real world setting
XGBoost (with G1/G2)      0.9200      0.9419      Mid-term prediction

- Failing students were consistently the hardest to classify across all models.
- Without grade features, absences, failures, and study time were the strongest predictors.
- Models with prior grades performed much better, but are only usable once the semester has started, highlighting a trade-off between accuracy and real-world applicability.

## Contributors

Justin Gurkin
Kaitlin Sloan
Manihams Suraparaju
Praneetha Voona
Joshua Grewe

