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

 # Final Report Write-up for Logistic Regression Baseline

To establish an initial performance benchmark for predicting student pass/fail outcomes, I developed a Logistic Regression baseline model using sociodemographic and behavioral features. I carefully preprocessed the dataset to avoid data leakage by excluding the intermediate grades G1 and G2, as well as the final grade G3, which was used only to create the binary pass/fail label (pass = G3 ≥ 10). To handle the mix of data types, I used a ColumnTransformer pipeline that applied one-hot encoding to categorical variables and standard scaling to numeric features. I then used an 80/20 stratified train-test split to maintain balanced class representation in both sets. The baseline Logistic Regression model, trained with a maximum of 500 iterations, performed strongly on the test set, achieving an F1 score of 0.9346 and a ROC-AUC of 0.9790. The confusion matrix showed just 7 misclassifications out of 79 test samples, including 22 true negatives, 50 true positives, 4 false positives, and 3 false negatives. Overall, these results show that even a relatively simple linear model can effectively capture the relationship between student characteristics and academic outcomes, making it a strong baseline for comparison with more complex ensemble methods.


# Final Report write up for Random Forest Classifier and Feature Importance
To model student success utilizing exclusively sociodemographic and behavioral traits, a Random Forest Classifier was implemented. To ensure the model did not artificially inflate its accuracy, intermediate and final grades (G1, G2, G3) were dropped from the feature set to prevent data leakage. The raw dataset was processed using a ColumnTransformer that applied standard scaling to numeric variables and one-hot encoding to categorical variables. To optimize model performance while preventing overfitting, hyperparameters were tuned using a Stratified 5-Fold Cross-Validation approach, optimizing for the F1-score. Evaluated on the holdout test set, the tuned Random Forest achieved an F1-score of [Insert F1 Score from Output] and an ROC-AUC of [Insert ROC-AUC from Output]. Finally, a Gini-based feature importance analysis was plotted to observe which traits most heavily drove the model's decisions. The analysis revealed that factors such as [Insert Top 3 Features from Bar Chart, e.g., past failures, absences, and parental education] were the most significant indicators in classifying whether a student would ultimately pass or fail.

# Final Report write up for Gradient Boosting Pipeline
A Gradient Boosting pipeline was developed using a stratified 5-fold cross validation to predict student pass/fail outcomes. There are two feature sets that were evaluted, one excluding grades G1, G2 and the other one included them. Without G1/G2, the model achieved an F1-score of 0.784 and a ROC-AUC of 0.625. This can indicate good classification performance but still weak seperation between the classes. The confusion matrix also shows a bias towards predicting pass with minimal fail predictions. When G1 and G2 were included, the performance improved greatly, with a F1 score of 0.92 and ROC-AUC of 0.942. This indicates that previous academic performance is predictable of outcomes. Though it brings about the issue of dependence of grading information that may not be available.
