# 🚀 Supervised Prediction Dashboard

A complete **Machine Learning Dashboard** built using Flask that allows users to upload datasets, train models, evaluate performance, and get predictions — all in one place.

---

## 📌 Features

* 📂 Upload CSV dataset
* 🔍 Automatic problem detection:

  * Classification
  * Regression
* ⚙️ Data preprocessing:

  * Missing value handling
  * Encoding categorical data
  * Feature scaling
* 🤖 Model training:

  * Logistic Regression
  * Linear Regression
  * Random Forest
* 📊 Evaluation Metrics:

  * Classification → Accuracy, F1 Score
  * Regression → R², MAE, MSE
* 🏆 Automatic best model selection
* 📈 Prediction output display

---

## 🛠️ Tech Stack

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 📁 Project Structure

```
ml_dashboard/
│
├── app.py
├── requirements.txt
├── model/
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── result.html
├── static/
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-sury228//
dashboard_supervised_prediction.git
cd ml-dashboard
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 🧠 How it works

1. Upload a dataset (CSV)
2. Select target variable
3. System detects problem type
4. Models are trained automatically
5. Performance metrics are calculated
6. Best model is recommended

---

## 🚀 Fufuygiguo inupomi-e Improvements

Hyperparameter tuning (GridSearchCV)

* Add more models (SVM, XGBoost)
* Interactive visualizations
* Model download (pickle)
* Deployment on cloud

---

## 💡 Use Case

* Beginners learning ML pipelines
* Rapid prototyping of ML models
* Demonstrating end-to-end ML systems

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Suryansh Jha**
Aspiring Data Scientist | Quant Developer
