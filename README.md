# ML Survival Predictor

An **end-to-end machine learning project** to predict passenger survival.  
This repository includes data preprocessing, model training, and a Streamlit app for interactive survival prediction.


---

## Project Overview

The **ML Survival Predictor** estimates the likelihood of passenger survival using a supervised learning model.  
The model is trained on historical data and predicts survival based on the following features:

- `Age` – Age of passenger  
- `Sex_encoded` – Encoded gender (Male = 1, Female = 0)  
- `Category_encoded` – Passenger or Crew (Passenger = 0, Crew = 1)  
- `Country_encoded` – Encoded nationality  

The project is fully **end-to-end**, including:

1. Preprocessing raw data  
2. Encoding categorical variables  
3. Training a machine learning classifier  
4. Exporting a pre-trained model (`best_model.pkl`)  
5. Interactive Streamlit app for manual or batch prediction  

---

## Features

- **Manual Entry:** Enter passenger details directly. Skipped fields use defaults.  
- **Batch Upload:** Upload CSV or Excel files with multiple records. Missing fields are auto-filled.  
- **Flexible Defaults:** Handles missing values gracefully.  
- **Interactive Streamlit App:** Visualize predictions and survival probabilities.  
- **Exportable Results:** Download predictions as CSV.

---


---

## Setup & Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/ml-survival-predictor.git
cd ml-survival-predictor
