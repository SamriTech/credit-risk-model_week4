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
