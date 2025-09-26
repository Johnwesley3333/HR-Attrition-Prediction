#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Project Title And Goal


# In[ ]:


# HR Attrition Prediction Project

**Goal:** Predict which employees are at risk of leaving the company and provide actionable HR recommendations.

**Dataset:** IBM HR Analytics — Employee Attrition & Performance (CSV in `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`)

**Skills Demonstrated:** Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, SMOTE, Feature Engineering, Model Evaluation, Random Forest, Logistic Regression, Data Visualization


# In[ ]:


Imports & Configuration


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
sns.set_style("whitegrid")


# In[ ]:


Load Dataset


# In[4]:


# Load CSV
df = pd.read_csv("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


# In[ ]:


Quick Data Checks


# In[5]:


# Shape & Info
print("Dataset shape:", df.shape)
print(df.info())

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Target value counts
print("Attrition counts:\n", df['Attrition'].value_counts())


# In[ ]:


Drop Irrelevant Columns


# In[6]:


drop_cols = ['EmployeeNumber','EmployeeCount','Over18','StandardHours']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
df.head()


# In[ ]:


EDA – Visualizations


# In[7]:


# Attrition distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Counts")
plt.show()

# Monthly Income vs Attrition
plt.figure(figsize=(8,4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Monthly Income vs Attrition")
plt.show()

# Correlation Heatmap
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


Preprocessing – Split X & y


# In[9]:


# Target & Features
y = df['Attrition'].map({'Yes':1,'No':0})
X = df.drop('Attrition', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


# In[ ]:


Preprocessing Pipeline


# In[11]:


import sklearn
print(sklearn.__version__)


# In[12]:


# Numeric & Categorical Features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Fit preprocessor on train
preprocessor.fit(X_train)

# Transform datasets
X_train_pre = preprocessor.transform(X_train)
X_test_pre = preprocessor.transform(X_test)

# Get feature names
ohe = preprocessor.named_transformers_['cat']
feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))


# In[ ]:


Handle Class Imbalance


# In[13]:


# SMOTE for balancing
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train_pre, y_train)

# Check balance
print("Class distribution after SMOTE:\n", pd.Series(y_train_res).value_counts())


# In[ ]:


Baseline Model – Logistic Regression


# In[14]:


lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_res, y_train_res)

# Predictions
y_pred_lr = lr.predict(X_test_pre)
y_prob_lr = lr.predict_proba(X_test_pre)[:,1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_prob_lr))


# In[ ]:


Stronger Model – Random Forest


# In[15]:


rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_res, y_train_res)

# Predictions
y_pred_rf = rf.predict(X_test_pre)
y_prob_rf = rf.predict_proba(X_test_pre)[:,1]

# Metrics
print(classification_report(y_test, y_pred_rf))
print("ROC AUC (RF):", roc_auc_score(y_test, y_prob_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


# In[ ]:


Feature Importance


# In[16]:


importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 15 Features:\n", feat_imp.head(15))

# Plot
feat_imp.head(15).sort_values().plot(kind='barh', figsize=(8,6), color='skyblue')
plt.title("Top 15 Feature Importances")
plt.show()


# In[ ]:


Save Model & Preprocessor


# In[17]:


import os
os.makedirs("../models", exist_ok=True)

joblib.dump(preprocessor, "../models/preprocessor.joblib")
joblib.dump(rf, "../models/random_forest_attrition.joblib")
print("Model and preprocessor saved in /models/")


# In[ ]:


Prediction Function Example


# In[18]:


def predict_employee_attrition(row_dict):
    row = pd.DataFrame([row_dict])
    Xp = preprocessor.transform(row)
    proba = rf.predict_proba(Xp)[0,1]
    pred = int(proba >= 0.5)
    return proba, pred

# Example
example = X_test.iloc[0].to_dict()
proba, pred = predict_employee_attrition(example)
print("Attrition probability:", proba, "Predicted class:", pred)


# In[ ]:


Summary & HR Recommendations (Markdown)


# In[ ]:


### Business Summary
- **Model used:** Random Forest Classifier
- **Key metrics:** Accuracy ~XX%, Recall ~XX%, ROC-AUC ~XX%
- **Top features affecting attrition:** OverTime, MonthlyIncome, JobSatisfaction, YearsAtCompany, JobRole

### HR Recommendations
1. Reduce overtime for at-risk employees.
2. Review compensation for employees in low-income brackets.
3. Improve job satisfaction through training and recognition programs.
4. Monitor employees with longer years at the company for burnout.
5. Tailor retention strategies based on job role.

