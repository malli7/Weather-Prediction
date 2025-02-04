# Logistic Regression for Weather Prediction

This repository contains a Jupyter Notebook demonstrating **Logistic Regression** for predicting rainfall using Australian weather data. It also includes preprocessed datasets and a trained model for easy experimentation.

## 📂 Repository Contents

- **Logistic_Regression_Weather_Prediction.ipynb**: Main notebook for training and evaluating the model.
- **aussie_rain_model.joblib**: Trained logistic regression model saved using `joblib`.
- **Dataset Files (Parquet Format)**:
  - `train_inputs.parquet`: Training set features.
  - `train_targets.parquet`: Training set labels (`RainTomorrow`).
  - `val_inputs.parquet`: Validation set features.
  - `val_targets.parquet`: Validation set labels.
  - `test_inputs.parquet`: Test set features.
  - `test_targets.parquet`: Test set labels.

## 📊 Dataset Details
The dataset consists of historical weather observations from multiple locations in Australia. It includes variables such as temperature, humidity, wind speed, and atmospheric pressure.

- **Number of Samples**: 140,787
- **Number of Features**: 23
- **Target Variable**: `RainTomorrow` (Binary: Yes/No)

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/weather-logistic-regression.git
cd weather-logistic-regression
```

### 2️⃣ Install Dependencies
Ensure you have Python and required libraries installed:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is not available, install these manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib plotly
```

### 3️⃣ Run the Notebook
Open the notebook and execute the cells:
```bash
jupyter notebook Logistic_Regression_Weather_Prediction.ipynb
```

## 🔍 Project Overview
The dataset contains various weather attributes from different locations in Australia. The goal is to predict whether it will rain tomorrow (`RainTomorrow`) using logistic regression.

### Key Steps in the Notebook:
- Data loading and cleaning
- Exploratory data analysis (EDA)
- Feature selection and preprocessing
- Model training and evaluation
- Saving and loading the trained model

## 📈 Model Performance Metrics
After training the logistic regression model, the following performance metrics were recorded on the test set:

- **Accuracy**: 85.4%
- **Precision**: 78.2%
- **Recall**: 72.5%
- **F1 Score**: 75.2%
- **AUC-ROC Score**: 0.82

## 🛠 Future Improvements
- Experimenting with more advanced models (e.g., Random Forest, Neural Networks)
- Hyperparameter tuning for better performance
- Feature engineering for better predictive power

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

