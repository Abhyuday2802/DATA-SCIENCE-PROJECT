# Customer Churn Prediction

An end-to-end final year project that predicts whether a telecom customer is likely to leave, compares three classification models, and provides an interactive retention dashboard.

## Project question

Can a customer's tenure, contract, services, and billing profile help identify customers who may churn? The project treats churn as a binary classification task and compares Logistic Regression, Decision Tree, and Random Forest models.

## Dataset

The project includes `data/telco_customer_churn.csv`, copied from the file supplied with the original notebook. It contains 7,043 customer records and 19 candidate input features plus the `Churn` target. `customerID` is removed because it identifies a row rather than describing a customer. Blank `TotalCharges` entries are interpreted as zero for customers whose tenure is zero.

The included data is for educational use. Results on this historical sample should not be treated as evidence of current customer behavior.

## What changed from the starting notebook

- Encoding and imputation are fitted inside each training fold by a scikit-learn pipeline. Validation rows do not influence preprocessing.
- A stratified 80/20 holdout set is kept separate from model selection.
- Candidate models are compared with five-fold stratified cross-validation on the training split. The selection score is mean ROC-AUC.
- Class weights handle the target imbalance during training; synthetic oversampling is not applied before cross-validation.
- The final holdout report includes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix.
- The saved artifact includes preprocessing and the classifier, so new profiles use the same transformations as training data.

## Setup

Use Python 3.10 or newer. From this project folder, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python src/train_model.py --data data/telco_customer_churn.csv --output .
```

Training writes:

- `artifacts/churn_pipeline.joblib` — fitted preprocessing and classifier.
- `reports/metrics.json` — cross-validation and holdout metrics.
- `reports/cv_model_comparison.png` — candidate ROC-AUC comparison.
- `reports/holdout_confusion_matrix.png` — holdout classification counts.
- `reports/feature_importance.csv` and `.png` — global model feature signals where supported.

## Run the prediction interface

The included trained model lets you open the app immediately. On Windows, double-click `run_app.bat`. On first launch it creates a local Python environment and installs the listed dependencies, then opens the app in your browser. Internet access is needed only for that first dependency installation.

The dashboard includes a portfolio overview, individual risk scoring with model-signal explanations, local CSV batch scoring with downloadable results, and a model-insights page with cross-validation and holdout performance.

To launch manually, install the dependencies and run:

```bash
streamlit run app.py
```

Enter a customer's service and billing profile to see the predicted class and model score. This interface is a classroom demonstration, not a production decision system.

## Notebook

Open `notebooks/Customer_Churn_Final_Year_Project.ipynb` from this project folder. It introduces the problem, inspects the data, visualizes the target and selected predictors, runs the training workflow, reviews evaluation plots, and demonstrates one prediction.

## Suggested final year project presentation

1. **Problem:** Customer churn reduces recurring revenue and makes retention planning harder.
2. **Objective:** Build and compare classifiers using customer service and billing attributes.
3. **Method:** Clean data, stratified split, fold-safe preprocessing, class-weighted model comparison, holdout evaluation.
4. **Results:** Present the generated `reports/metrics.json` and plots. Emphasize recall and PR-AUC alongside accuracy because the churn class is smaller.
5. **Limitations:** This is a single historical dataset, the model scores are not calibrated, and the evaluation does not estimate the impact or cost-effectiveness of retention actions.
6. **Future work:** Test on newer data, calibrate probabilities, tune the operating threshold using business costs, monitor performance over time, and add privacy and fairness review before any real deployment.

## Reproducibility

The training script fixes the random seed at 42 and records the split size, class counts, cross-validation results, chosen model, and test metrics. For a formal submission, include the generated metrics and plots in the report and note the Python/package versions used.
