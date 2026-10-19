# Employee Attrition Analytics Platform

An end-to-end HR analytics project on the IBM HR Analytics dataset (1,470 employees, 16.1% attrition).
It compares neural networks with classical, Bayesian and discriminant models, selects the best attrition model,
deploys it on IBM Watson Machine Learning as a REST endpoint, and feeds an HR analytics dashboard.

## Project Goals
- Compare **Neural Networks** with classical ML
- Explore **Bayesian classification** (Gaussian Naive Bayes)
- Apply **Discriminant Analysis** (LDA / QDA)
- Deploy the best model with **IBM Watson Studio and Watson Machine Learning** (REST scoring endpoint)
- Automate retraining and redeployment with **Watson Studio Jobs**
- Build an **HR analytics dashboard** and an **executive presentation**

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   └── Employee_Attrition_Analytics_Platform.ipynb   # analysis + report + Watson deployment
├── model/
│   └── attrition_model.pkl                           # best model (scikit-learn pipeline)
├── figures/                                          # charts used in the report
├── results/
│   └── model_comparison.csv                          # metrics for all models
├── dashboard_data/
│   ├── employee_risk.csv                             # risk score + category per employee
│   └── dashboard_data.json                           # KPIs, department stats, model metrics
├── dashboard/                                        # interactive HR dashboard
├── reports/
│   └── Final_Technical_Report.docx                   # includes REST API documentation
└── presentation/                                     # executive presentation
```

## Methods
| Family | Models | Notes |
|---|---|---|
| Classical ML | Logistic Regression, Random Forest, XGBoost | class weights / `scale_pos_weight` for imbalance |
| Neural Network | MLP (64-32 hidden units) | regularised, early stopping, oversampled training data |
| Bayesian | Gaussian Naive Bayes | prior and posterior attrition probability |
| Discriminant Analysis | LDA, QDA | numeric features only (assumes continuous, normal inputs) |

- Preprocessing: dropped constant/ID columns, scaled numeric features, one-hot encoded categoricals, stratified 80/20 split
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, plus 5-fold stratified cross-validation
- Model selection: highest cross-validated ROC-AUC

## Key Results
| Family | Best model | CV ROC-AUC |
|---|---|---|
| Classical ML | Logistic Regression | 0.831 |
| Neural Network | MLP | 0.812 |
| Bayesian | Gaussian Naive Bayes | 0.764 |
| Discriminant | LDA | 0.753 |

- **Selected model:** Logistic Regression. The neural network came close but did not outperform it on this small tabular dataset.
- **Top drivers of attrition:** overtime, years since last promotion, number of companies worked at.
- **Risk categories** (Low / Medium / High) separate leavers well: actual attrition is about 4% in Low, 14% in Medium and 45% in High risk.

Full results are in `results/model_comparison.csv` and the technical report.

## How to Run
1. Open `notebooks/Employee_Attrition_Analytics_Platform.ipynb` in [Google Colab](https://colab.research.google.com).
2. Click **Runtime > Run all** and upload `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` when asked.
3. The notebook trains and compares all models, saves the best one, exports the dashboard data and generates the technical report.
4. A zip with all outputs downloads automatically.

## Deployment on IBM Watson Machine Learning
The last section of the notebook deploys the best model as an online REST endpoint.

1. Create an IBM Cloud account with **Watson Studio** and **Watson Machine Learning** services.
2. Create a Watson Studio project, a **Deployment Space** and an **API key**.
3. In the notebook set `DEPLOY_TO_WATSON = True` and fill in `SPACE_ID` (and the API key when prompted).
4. Run the notebook. It applies a quality gate (test ROC-AUC >= 0.78), stores the model, creates or updates the deployment and tests scoring.

**Scoring endpoint:** `<paste your scoring URL here after deployment>`

### REST API (summary)
```
POST https://<region>.ml.cloud.ibm.com/ml/v4/deployments/<deployment_id>/predictions?version=2021-05-01
Authorization: Bearer <IAM token>
Content-Type: application/json

{"input_data": [{"fields": ["Age", "BusinessTravel", ...], "values": [[41, "Travel_Rarely", ...]]}]}
```
The response contains the predicted class (1 = likely to leave) and the probabilities `[P(stay), P(leave)]`.
Full documentation and the complete field list are in Appendix A of the technical report.

### Automated retraining
Run the notebook as a **Watson Studio Job** on a schedule (Jobs > Create a job > Schedule). Each run retrains the models, checks the quality gate,
and updates the existing deployment so the endpoint URL stays the same.

## Dashboard
The HR dashboard (in `dashboard/`) shows attrition prediction, attrition by department, employee risk categories, workforce demographics,
model performance metrics and HR KPIs (retention rate, attrition rate, department-wise trends). It is powered by the files in `dashboard_data/`.

## Limitations
- Small, fictional dataset; results may not transfer to a real company.
- Feature importance shows association, not cause.
- Predictions should support supportive HR conversations, never punitive decisions. Sensitive attributes need fairness checks before real use.

## Tech Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, python-docx, IBM Watson Studio, IBM Watson Machine Learning, Google Colab.
