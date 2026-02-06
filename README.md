# Credit Risk Probability Model

## Business Context
Banks must assess the likelihood that a customer will default on a loan. 
In this project, we build a credit risk probability model using alternative transaction data, 
aligned with Basel II regulatory principles.

## Objective
- Predict customer credit risk using transactional behavior
- Build an interpretable and auditable model suitable for banking use
- Deploy the model as an API for real-time risk assessment

## Why Interpretability Matters
Regulatory frameworks such as Basel II require banks to explain credit decisions.
Therefore, interpretable models like Logistic Regression are preferred over black-box models
when transparency and accountability are critical.

## Dataset Challenge
The dataset does not include a direct "default" label.
To overcome this, we engineer a proxy target variable using customer transaction behavior
through Recency, Frequency, and Monetary (RFM) analysis.

## Methodology
1. **Exploratory Data Analysis (EDA):** Analyze distributions, correlations, and outliers.  
2. **Feature Engineering:** Aggregate transaction metrics, extract time-based features, encode categorical variables, normalize data, and apply WoE transformations.  
3. **Proxy Target Creation:** Cluster customers with RFM metrics to define `is_high_risk`.  
4. **Model Training:** Train models including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting; tune hyperparameters; track experiments with MLflow.  
5. **Deployment:** Build a FastAPI API for predictions, containerize with Docker, and automate testing with CI/CD pipelines.

