# Customer Churn Prediction with Multilayer Perceptron (MLP) using PySpark

## Project Overview

In today's competitive markets, customer retention is more critical than ever.  
This project leverages **deep learning** techniques to **predict customer churn** using **PySpark's Multilayer Perceptron Classifier (MLPClassifier)**, providing actionable insights that can help businesses proactively manage and reduce churn rates.

## Objective

- Build a predictive model that identifies customers who are likely to churn.
- Apply deep learning methods to large-scale data using distributed computing (PySpark).
- Optimize model performance through hyperparameter tuning.

## Business Problem

Customer churn directly impacts revenue and growth.  
Early identification of customers at risk allows companies to implement targeted retention strategies such as personalized offers, loyalty programs, or customer service improvements.

---

## Dataset Description

The dataset includes customer information across various dimensions:

- **Usage behavior** (call minutes, internet usage, etc.)
- **Demographics** (age, gender, income level)
- **Subscription details** (contract type, monthly charges, tenure)

---

## Project Pipeline

### 1. Data Preprocessing

- Handle missing values appropriately.
- Encode categorical variables using one-hot encoding or label encoding.
- Scale numerical features to standardize input for neural network training.

### 2. Model Building

- Train a **Multilayer Perceptron Classifier** where the target variable is churn (1 = churned, 0 = retained).
- Experiment with various network architectures:
  - Vary the **number of hidden layers**.
  - Adjust the **number of neurons** in each layer.
  - Tune **max_iter** (maximum number of iterations for convergence).

### 3. Model Evaluation

- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC (Area Under the Curve)

- Visualization:
  - Confusion Matrix
  - ROC Curve

- Analyze the impact of different hyperparameters on model performance.

---

## Results

After optimizing key hyperparameters (especially **max_iter** and **hidden layers**), the model showed significant improvements:

| Metric    | Before Optimization | After Optimization |
|-----------|----------------------|--------------------|
| Accuracy  | 0.91                 | 0.96               |
| Precision | 0.86                 | 0.96               |
| Recall    | 0.91                 | 0.96               |
| F1 Score  | 0.87                 | 0.96               |
| AUC       | 0.68                 | 0.92               |

The **AUC** saw the largest gain, highlighting the model's improved ability to separate churners from non-churners.

---

## Key Insights

- **Hyperparameter optimization** (especially increasing max_iter and refining network architecture) plays a crucial role in boosting model performance.
- **Precision and Recall improvements** suggest the model became much better at minimizing false positives and capturing actual churners — critical for retention strategies.
- A significant jump in **AUC** indicates that the model can now much better distinguish between customers who are likely to churn and those who are not.

---

## Business Recommendations

Based on the model outcomes:

1. **Early Warning System**:  
   Implement real-time churn prediction scoring for customers based on updated behavioral data, allowing marketing teams to act proactively.

2. **Targeted Retention Campaigns**:  
   Focus retention efforts (such as personalized promotions, loyalty programs, or service upgrades) specifically on high-risk customers identified by the model.

3. **Continuous Monitoring**:  
   Set up periodic retraining and recalibration of the model as customer behavior and market conditions change, ensuring long-term model effectiveness.

4. **Further Segmentation**:  
   Segment churners into actionable groups (e.g., high-value vs. low-value customers) to prioritize retention efforts where the financial impact is greatest.

---

## Technologies Used

- **PySpark** (for scalable data processing and model training)
- **Python** (data wrangling, visualization)
- **MLlib** (Spark's machine learning library)
- **Matplotlib / Seaborn** (for visualization)

---

## Next Steps

- Explore more advanced deep learning models (e.g., CNNs for sequential customer activity data).
- Integrate model predictions into CRM systems for live churn monitoring.
- Perform feature importance analysis to identify key drivers of churn.

---

> **Note**:  
> This project demonstrates how deep learning models like MLPClassifier can be effectively applied in real-world business contexts to drive strategic decision-making.

