# 📊 Automated Data Analyst System

An end-to-end **Automated Data Analyst System** that performs dataset understanding, statistical analysis, visual exploration, and optional machine learning — all through an interactive **Streamlit web application**.

This project is designed as a data analyst system, not just a personal tool, and demonstrates how automated analytics platforms work in real-world scenarios.

## 🚀 Features

### 🔍 Data Understanding
- Upload CSV datasets
- Automatic schema detection
- Identifies numerical, categorical, and missing-value columns

### 📈 Statistical Analysis
- Descriptive statistics (mean, median, std, min, max)
- Automated summaries for numerical features

### 💡 Insight Generation
- Rule-based insights generated from schema and statistics
- Detects data imbalance, skewness, and missing values

### 📊 Visual Analytics
- Distribution plots for numerical columns
- Category count plots for categorical columns
- Interactive column selection via UI

### 🤖 Machine Learning (Optional)
- **Automatic problem type detection**: Regression / Classification
- **Built-in models**:
  - Linear Regression
  - Logistic Regression
- **Performance metrics**:
  - Regression: RMSE, R²
  - Classification: Accuracy, Classification Report

*The ML module runs only when a target column is selected, keeping the system flexible and safe for pure analysis use cases.*

## 🧱 Project Structure

data-analyst-system/
│

├── app.py                     # Main Streamlit application

├── requirements.txt           # Project dependencies
│

├── core/

│   ├── data_loader.py         # Data loading logic

│   ├── schema_analyzer.py     # Dataset schema detection

│   ├── analysis_engine.py     # Statistical analysis engine

│   ├── insights.py            # Insight generation module

│   ├── visualizer.py          # Visualization utilities

│   └── ml_engine.py           # Machine learning module

│

├── utils/

│   └── helpers.py             # Shared helper functions



## 🛠 Tech Stack
- **Python** 3.10+
- **Streamlit** – Interactive web UI
- **Pandas & NumPy** – Data processing
- **Matplotlib / Seaborn** – Visualizations
- **Scikit-learn** – Machine learning models

## ⚙️ Installation (Local)

1. **Clone the repository**
   git clone https://github.com/<your-username>/data-analyst-system.git
   cd data-analyst-system

2. **Create and activate virtual environment**
  python -m venv venv
  source venv/bin/activate   # Windows: venv\Scripts\activate
3. **Install dependencies**
  pip install -r requirements.txt
4. **Run the application**
  streamlit run app.py

## 🌐 Deployment (Streamlit Community Cloud)

This project is deployed using **Streamlit Community Cloud** – *Your fully hosted app is ready to share in under a minute!*

**Ridiculously easy deployment:**

1. **Sign in with GitHub**
2. **Pick a repo, branch, and file**
3. **Click Deploy!**

**Detailed Steps:**
1. Push the project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Select:
   - **Branch**: `main`
   - **App file**: `app.py`
5. Click **Deploy**

> 💡 *Then any time you do a `git push`, your app will update immediately!*

The app will be live within minutes. ✨

## 📌 Use Cases
- Automated Exploratory Data Analysis (EDA)
- Academic and internship projects
- Business data understanding
- Interview-ready demonstration system
- Base platform for future AI-powered analytics

## 🔮 Future Enhancements
- AutoML integration (RandomForest, XGBoost)
- Cross-validation and model comparison
- Feature importance explanations
- Dataset profiling reports
- LLM-based natural language insights
- Support for Excel and JSON files

## 👤 Author
**Ittyavira C Abraham**  
*MCA – AI & Machine Learning*  
Focused on building intelligent, real-world data systems.

## ⭐ Acknowledgements
- Streamlit Community
- Scikit-learn
- Open-source Python ecosystem
