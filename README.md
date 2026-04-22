# ML-325-project
STUDENT PERFORMANCE PREDICTION
Environments -
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
shap
jupyter


Kaitlin Sloan — Logistic Regression Pipeline
Load the raw dataset herself
Preprocess it (OHE, binary mapping, StandardScaler)
Train a Logistic Regression baseline with and without G1/G2
Evaluate: F1, ROC-AUC, confusion matrix
Write up her results in a short paragraph for the final report
Push her own notebook to the repo


Praneetha Voona — Random Forest Pipeline
Load and preprocess the raw dataset herself
Train Random Forest with stratified 5-fold CV and hyperparameter tuning
Evaluate: F1, ROC-AUC, confusion matrix
Plot feature importances (Gini-based)
Write up her results in a short paragraph for the final report
Push her own notebook to the repo


Joshua Grewe — Gradient Boosting Pipeline
Load and preprocess the raw dataset himself
Train XGBoost or LightGBM with stratified 5-fold CV
Evaluate: F1, ROC-AUC, confusion matrix
Compare results with vs. without G1/G2
Write up his results in a short paragraph for the final report
Push his own notebook to the repo


Justin Gurkin — SHAP Fairness Analysis
Load and preprocess the raw dataset himself
Train a Random Forest (just enough to get SHAP values — doesn't need to be tuned)
Run SHAP analysis on all features
Specifically investigate parental education: does removing it change predictions significantly?
Write up the fairness/bias findings section of the final report
Push his own notebook to the repo


Manihams Suraparaju — Ensemble + Final Report
Load and preprocess the raw dataset himself
Train a soft voting ensemble (LR + RF + XGBoost) using sklearn's VotingClassifier
Evaluate: F1, ROC-AUC, confusion matrix
Compile the full 4-page IEEE final report — pull the written paragraphs from everyone's notebooks and assemble Abstract, Methodology, Results, Conclusion, Workload Distribution
Clean up the repo and add requirements.txt
Push his own notebook + the final report
