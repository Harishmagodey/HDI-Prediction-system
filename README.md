# HDI-Prediction-system
Machine Learning based HDI Prediction using Flask

## Overview

This project predicts the Human Development Index (HDI) using Machine Learning based on three key indicators:

- Life Expectancy
- Expected Years of Schooling
- Gross National Income (GNI)

The application is built using Flask and Scikit-learn and provides an interactive web interface for predicting HDI values.

---

## Project Workflow

```
HDI Dataset (CSV)
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Model Training
        ↓
Save Model (.pkl)
        ↓
Flask Web Application
        ↓
User Inputs
        ↓
HDI Prediction
        ↓
Display Result
```

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- Joblib

---

## Project Structure

```
HDI-Prediction-system/
│
├── app.py
├── train_model.py
├── README.md
│
├── dataset/
│   └── HDI.csv
│
├── model/
│   ├── hdi_model.pkl
│   └── imputer.pkl
│
├── static/
│   ├── style.css
│   └── images/
│       └── background.jpg
│
└── templates/
    └── index.html
```

---

## How to Run

1. Clone the repository.

2. Install the required packages:

```bash
pip install flask pandas numpy scikit-learn joblib
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## Features

- Interactive Flask Web Application
- Machine Learning-based HDI Prediction
- Modern Responsive User Interface
- Real-time Prediction
- Easy to Use

---

## Author

Harishma Godey
