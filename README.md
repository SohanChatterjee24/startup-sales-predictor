# 🚀 Kickstarter Campaign Success Prediction

Python(https://www.python.org/)
Jupyter(https://jupyter.org/)
XGBoost(https://xgboost.readthedocs.io/)

> Machine learning project to predict Kickstarter campaign success using EDA, preprocessing, modeling, and evaluation.  
> **Best Model:** XGBoost with ~88% accuracy.

---

## 📂 Project Structure
data/
│── processed/ # Processed datasets
│ ├── 01_eda_cleaned.csv
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ ├── y_test.csv
│
│── raw/ # Original dataset
│ ├── ks-projects-201801.csv
│
notebooks/ # Jupyter Notebooks for each step
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_modeling.ipynb
│ ├── 04_evaluation.ipynb
│
outputs/ # Project outputs
│ ├── Final-Report.pdf
│ ├── model_results.csv
│
src/ # Python scripts (replica of notebooks)
│ ├── 01_eda.py
│ ├── 02_preprocessing.py
│ ├── 03_modeling.py
│ ├── 04_evaluation.py

🚀 Usage
Run in order:
01_eda.ipynb
02_preprocessing.ipynb
03_modeling.ipynb
04_evaluation.ipynb

🐍 Run Python Scripts
src/01_eda.py
src/02_preprocessing.py
src/03_modeling.py
src/04_evaluation.py

📊 Workflow
EDA – Data inspection, cleaning, visualizations.
Preprocessing – Handling missing values, encoding, scaling.
Modeling – Logistic Regression, Random Forest, XGBoost.
Evaluation – Accuracy, Precision, Recall, F1, ROC AUC.
