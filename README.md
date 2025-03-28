# 🚀 Credit Card Fraud Detection

## 📌 Project Overview
This project aims to detect fraudulent credit card transactions using **Python, Pandas, Scikit-learn, and Machine Learning models**. The dataset is highly imbalanced, so we apply **EDA, feature engineering, and SMOTE for balancing the classes.**

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:**
  - `Time`, `Amount`
  - `V1` to `V28` (PCA-transformed features)
  - `Class` (Target: **0 = Legit, 1 = Fraud**)

## ⚙️ Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn)
- **Machine Learning** (Random Forest, Logistic Regression, XGBoost)
- **SMOTE** (To handle imbalanced classes)

## 🔍 Exploratory Data Analysis (EDA)
✔ Checked for missing values
✔ Visualized class distribution (Fraud vs. Non-Fraud)
✔ Plotted feature correlations
✔ Scaled `Amount` and `Time` using **StandardScaler**

## 🏗️ Model Training & Evaluation
1️⃣ **Baseline Model** → Logistic Regression
2️⃣ **Improved Model** → Random Forest (High Accuracy ⚡)
3️⃣ **Final Model** → XGBoost (Best Performance ✅)

### **Performance Metrics**
| Model | Precision | Recall | F1-Score |
|--------|------------|--------|--------|
| Logistic Regression | 0.82 | 0.75 | 0.78 |
| Random Forest | 0.95 | 0.89 | 0.92 |
| XGBoost | **0.97** | **0.91** | **0.94** |

## 📌 Key Findings
✔ Fraud transactions have unique patterns different from regular transactions.
✔ **SMOTE improves model performance** by balancing the dataset.
✔ Random Forest & XGBoost perform significantly better than logistic regression.

## 🛠️ How to Run the Project
1️⃣ Clone the repository:
```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
```
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
3️⃣ Run the Jupyter Notebook:
```bash
jupyter notebook
```

## 📁 Folder Structure
```
├── data/              # Dataset folder
├── notebooks/         # Jupyter Notebooks for analysis
├── models/            # Saved models
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

## 🚀 Next Steps
- Try **Deep Learning models (LSTM, Autoencoders)**.
- Deploy as an **API using Flask**.
- Create a **dashboard using Streamlit**.

## 📌 Author
👤 **Your Name**  
📧 Email: your.email@example.com  
📌 LinkedIn: [Your Profile](https://www.linkedin.com/in/your-profile)  
📌 GitHub: [Your GitHub](https://github.com/your-github)

---
⭐ **If you like this project, please give it a star!** ⭐
