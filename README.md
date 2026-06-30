# ☕ Monday Coffee — Customer Churn Prediction



---

## Objective

This project is a direct extension of the [Monday Coffee Expansion Analysis](https://github.com/sohaillansarii/Monday-Coffee-Expansion), where SQL and Power BI were used to identify the best cities for expansion and understand customer behavior across markets like Pune, Delhi, and Jaipur. Having established where to grow, the next challenge was **retention** — identifying which customers are likely to churn before they actually leave.

This project builds a **XGBoost machine learning model** trained on Monday Coffee's customer behavioral data to predict churn (0 = Stays, 1 = Churns). The model is deployed as a **live REST API using FastAPI on Railway**, making it accessible to any downstream application or dashboard in real time — no UI needed, just plug and predict.

---

## 🚀 Live API

**Base URL:** `https://monday-coffee-churn-prediction.up.railway.app/`

**Swagger Docs URL:** `https://monday-coffee-churn-prediction.up.railway.app/docs`


---

### Project Brief

The dataset was cleaned and preprocessed to handle missing values, outliers, and inconsistent entries before any modeling began. A scikit-learn Pipeline with ColumnTransformer was used throughout to ensure zero data leakage — all transformations are fit only on training data and applied to the test set at inference time. The model was validated on a held-out test set of 1,200 records, achieving 99% accuracy with 100% recall on churned customers, meaning no at-risk customer was missed. Feature importance was also visualized in the notebook to understand which signals — like **satisfaction_score**, **engagement_score**, and **total_spend** — drive churn the most.

### Features Used

| Feature | Description |
|---|---|
| `account_age_months` | How long the customer has been with Monday Coffee |
| `avg_order_value` | Average spend per order |
| `total_orders` | Total number of orders placed |
| `discount_usage_rate` | How frequently discounts are used |
| `return_rate` | Product return frequency |
| `customer_support_tickets` | Number of support interactions |
| `loyalty_member` | Whether the customer is in the loyalty program |
| `browsing_frequency_per_week` | Weekly app/site visits |
| `cart_abandonment_rate` | Frequency of abandoned carts |
| `product_review_score_avg` | Average product rating given |
| `engagement_score` | Overall engagement with the brand |
| `satisfaction_score` | Customer satisfaction rating |
| `price_sensitivity_index` | Sensitivity to price changes |
| `total_spend` | Cumulative spend over lifetime |


---

##  Model Performance

```
              precision    recall  f1-score   support

  Stayed (0)       1.00      0.99      1.00       990
 Churned (1)       0.96      1.00      0.98       210

    accuracy                           0.99      1200
   macro avg       0.98      0.99      0.99      1200
weighted avg       0.99      0.99      0.99      1200
```

- **99% overall accuracy** on the held-out test set
- **100% recall on churned customers** — the model catches every single at-risk customer
- **Zero data leakage** — achieved through Pipeline + ColumnTransformer design

---

##  Tech Stack

- **Modeling:** XGBoost, scikit-learn Pipeline, ColumnTransformer
- **API:** FastAPI + Uvicorn
- **Deployment:** Railway (Dockerized)
- **Notebook:** Jupyter (EDA, preprocessing, feature importance, model training)

---




## Sample API Request

```python
import requests

response = requests.post(
    "https://monday-coffee-churn-prediction.up.railway.app/predict",
    json={
        "account_age_months": 24,
        "avg_order_value": 300,
        "total_orders": 10,
        "discount_usage_rate": 0.3,
        "return_rate": 0.1,
        "customer_support_tickets": 2,
        "loyalty_member": 1,
        "browsing_frequency_per_week": 3,
        "cart_abandonment_rate": 0.2,
        "product_review_score_avg": 4.2,
        "engagement_score": 3.5,
        "satisfaction_score": 4,
        "price_sensitivity_index": 0.5,
        "total_spend": 3000
    }
)
print(response.json())
```

**Response:**
```json
{'churn_prediction': 1, 
'churn_label': 'Will Churn', 
'churn_probability': 0.9994}
```

---

## 🔗 Related Project

This project builds on top of → [Monday Coffee Expansion (SQL + Power BI)](https://github.com/sohaillansarii/Monday-Coffee-Expansion)

## AUTHOR
**Sohail Ansari**