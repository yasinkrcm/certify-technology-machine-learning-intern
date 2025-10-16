# Certify Technology Machine Learning Internship — Projects

This repository contains complete, runnable solutions and outputs for the internship tasks. All plots are saved automatically into the `photos/` directory when you run the notebook.

## How to Run
1. Open `Compulsory_Internship_Tasks.ipynb` in Jupyter/VS Code.
2. Run all cells sequentially. Datasets are downloaded from public mirrors and cached under `data/`.
3. Generated figures will appear under `photos/` and are referenced below.

## Contents

### Level 1 — Coding Challenges
- Linear Regression (California Housing): prints MSE and R²; scatter plot of Actual vs Predicted.
- Decision Tree (Iris): accuracy, confusion matrix, and tree visualization.
  - Images: `photos/iris_confusion_matrix.png`, `photos/iris_tree.png`

### Level 1 — Spam Email Classifier (Compulsory)
- TF-IDF + MultinomialNB; metrics printed (accuracy, precision, recall, F1)
  - Image: `photos/spam_confusion_matrix.png`

### Level 2 — Insurance Charges Prediction
- Models: Linear Regression, RandomForest, (optional) XGBoost
- Metrics: MSE, R² (printed)
- Visualizations:
  - Correlation heatmap: `photos/insurance_corr_heatmap.png`
  - Actual vs Predicted (top models): `photos/insurance_actual_vs_pred_<ModelName>.png`
  - Feature importances: `photos/insurance_feature_importances_<ModelName>.png`

### Level 2 — Retail Customer Segmentation (K-Means)
- Elbow method then K-Means; PCA 2D clusters
  - Images: `photos/segmentation_elbow.png`, `photos/segmentation_pca_clusters.png`

### Level 3 — Credit Card Fraud Detection
- Models: Logistic Regression, RandomForest; robust preprocessing
- Metrics: ROC-AUC and PR-AUC (printed)
- Visualizations:
  - ROC curves: `photos/fraud_roc_curves.png`
  - Precision-Recall curves: `photos/fraud_pr_curves.png`

## Notes
- If a mirror is down, cells fall back to alternative sources (or generate a small synthetic dataset for segmentation) so the notebook remains runnable offline.
- All images are regenerated on each run; feel free to delete `photos/` and re-run.


