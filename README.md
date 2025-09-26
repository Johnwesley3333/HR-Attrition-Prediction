# 🧑‍💼 HR Attrition Prediction & Dashboard

*A comprehensive analysis and prediction of employee attrition using Python and Machine Learning.*

---

## 🌟 Overview
This project analyzes **employee data** to predict **attrition probability** and helps HR teams make **data-driven retention decisions**. It includes a **Streamlit dashboard** for interactive prediction.  

**Key Objectives:**  
- Predict employee attrition probability using historical data.  
- Identify top factors affecting employee turnover.  
- Build an interactive **Streamlit dashboard** for HR decision-making.  
- Showcase a portfolio-ready ML project with deployment.  

---

## 📊 Dataset Overview
The dataset contains **employee-level data** with attributes such as:  
- **Age** – Employee age  
- **MonthlyIncome** – Monthly salary of the employee  
- **YearsAtCompany** – Tenure at the company  
- **DistanceFromHome** – Distance of employee’s residence from office  
- **OverTime** – Whether employee works overtime (Yes/No)  
- **JobRole** – Role of the employee in the company  
- **JobSatisfaction** – Job satisfaction level (1=Low, 4=High)  
- **Attrition** – Target variable (Yes/No)  

**Size:** Approximately X rows × Y columns (replace with actual numbers).

---

## 🎯 Project Workflow
✅ **Data Cleaning & Preprocessing** – Handle missing values, encode categorical features, scale numeric features.  
✅ **Feature Selection** – Focus on **top 7 features** for model simplicity and app usability.  
✅ **Exploratory Data Analysis (EDA)** – Visualize key patterns affecting attrition.  
✅ **Model Training** – Train **Random Forest Classifier** to predict attrition probability.  
✅ **Model Deployment** – Build an interactive **Streamlit app** for predictions.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 🐍  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit  
- **Tools:** Jupyter Notebook, GitHub  
- **Dashboarding:** Streamlit interactive dashboard  
- **ML Models:** Random Forest Classifier  

---

## 📂 Project Structure
```
HR-Attrition-Project/
pp/
│ └── app.py # Streamlit dashboard for interactive prediction
├── data/
│ └── employee_data.csv # Dataset file
├── models/
│ ├── preprocessor_top.joblib
│ └── random_forest_top.joblib
├── notebooks/
│ └── HR_Attrition_Notebook.ipynb
├── requirements.txt # Project dependencies
├── .gitignore
└── README.md           
```


---

## 🚀 Installation & Setup
1️⃣ **Clone the repository**  

```bash
git clone https://github.com/YOUR_USERNAME/HR-Attrition-Prediction.git
cd HR-Attrition-Prediction

```
2️⃣ Install dependencies
```
pip install -r requirements.txt
```
3️⃣ Run the Streamlit dashboard
```
streamlit run app/app.py

```
4️⃣ Open your browser at http://localhost:8501 to explore the dashboard.
## 📈 **Key Insights & Forecasting**
### **Exploratory Data Analysis (EDA)**
- **Attrition by Age & Tenure** – Younger employees with shorter tenure show higher attrition probability..
- **Job Role & Satisfaction Impact** – Certain roles and lower satisfaction correlate with higher attrition.
- **OverTime Analysis** – Employees working overtime are more likely to leave.

### **Sales Forecasting Results (ARIMA Model)**
| Feature        | Input Value |
|-------------|--------------------|
| Age             | 30 |
| MonthlyIncome   | 5000 |
| YearsAtCompany   | 3 |
| DistanceFromHome    | 10 |
| OverTime    | Yes |
| Job Role    | Sales Executive |
| JobSatisfaction  | 3 |
Predicted Attrition Probability: 0.72
redicted Class: Yes

## 📉 **Conclusion**
This project provides a **data-driven approach** to predict employee attrition. The **Random Forest model** and interactive dashboard help HR teams identify at-risk employees and improve retention strategies.

## 🤝 **Contributions**
💡 Open to improvements! Feel free to:
1. Fork the repo  
2. Create a new branch (`feature-branch`)  
3. Make changes & submit a PR  



## 📩 **Connect with Me**
📧 **Email:** [johnwesleykolasanakoti@gmail.com](mailto:johnwesleykolasanakoti@gmail.com)  
🌐 **Portfolio:** [K-John Wesley Portfolio]()  
💼 **LinkedIn:** [K-John Wesley](www.linkedin.com/in/john-wesley-794125284)  
👨‍💻 **GitHub:** [K- John Wesley](https://github.com/Johnwesley3333)  

⭐ **If you find this project useful, drop a star!** 🚀
