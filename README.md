# Banff Traffic Management – Machine Learning Prediction System

A machine learning project that predicts hourly parking occupancy in Banff and identifies when parking units will reach over 90% capacity. This model supports planning, staffing, and congestion management for the Town of Banff.  
The project includes data cleaning, feature engineering, EDA, regression and classification models, SHAP explainability, and a deployed Streamlit MVP app.

---

## 📌 Project Objectives
- Predict hourly parking occupancy using regression models  
- Identify “full” periods (>90% occupancy) with classification models  
- Explore seasonal, hourly, and visitor trends in Banff  
- Build a real-time prediction app using Streamlit  
- Provide explainability using SHAP values  

---

## 📂 Dataset Summary
This project uses several real datasets from the Banff Traffic System:

- Parking sessions 2023  
- Parking sessions 2024  
- Visits dataset  
- Hourly traffic volumes  
- Route travel times  
- Weather + seasonal data  

All datasets were cleaned, merged, and transformed into a single processed table used for modeling.

---

## 📊 Exploratory Data Analysis (EDA)
Key insights from EDA:

- Afternoon and weekend peaks show highest occupancy  
- Summer months have significantly higher demand  
- Strong correlations exist between visits, time, and occupancy  
- Trend plots, heatmaps, and histograms reveal clear seasonal patterns  

Example graphs (add your images):
- Occupancy trend over time  
- Visits distribution  
- Correlation heatmap  
- Seasonal comparison charts  

---

## 🛠️ Feature Engineering
Features created to enhance model performance:

- Lag features (lag_1, lag_24)  
- Rolling averages (rolling_7, rolling_14)  
- Hour of day  
- Day of week  
- Weekend indicator  
- Seasonal indicator  
- Visit counts  
- Travel delays  
- Weather conditions  
- `Is_Full` classification label (>90% occupancy)  

These features capture time patterns, peaks, seasonality, and visitor flow.

---

## 🤖 Machine Learning Models

### **Regression: Predict Hourly Occupancy**
Models trained:
- XGBoost Regressor  
- LightGBM Regressor  
- Linear Regression baseline  

Evaluation metrics:
- MAE ~78  
- RMSE ~180  

### **Classification: Predict Is_Full (>90%)**
- LightGBM Classifier  
- High recall on full periods  
- Useful as an early warning system  

---

## 🧠 Explainability (SHAP)
SHAP analysis was performed to understand model behavior.

Most important features:
- Hour of day  
- Visit counts  
- Seasonal effects  
- Lag features  

SHAP plots included:
- Bar importance plot  
- Beeswarm plot  

---

## 📱 Streamlit MVP App
A working MVP app was created to:

- Show predicted occupancy  
- Indicate if a unit will be full  
- Display SHAP explanations  
- Provide basic visualization  

### Add your app link here:
👉 **Streamlit App:** _your-app-link-here_  

---

## 📁 Folder Structure
'''
project/
│── data/
│ ├── raw/
│ └── processed/
│── notebooks/
│── src/
│── models/
│── app/
│── README.md
│── requirements.txt
'''

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

🔮 Future Improvements

Add live weather and traffic API integration

Deploy full version with database support

Build multi-location prediction dashboards

Improve classification recall for high-demand periods

🙋‍♂️ Author

Akshit Bhandari
Machine Learning Student – NorQuest College
GitHub: https://github.com/AKSHIT224
