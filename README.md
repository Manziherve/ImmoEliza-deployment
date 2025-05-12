# 🏠 Belgian Immo Price Prediction App

This is a machine learning web application that predicts the price of real estate in Belgium based on property features such as type, location, surface area, and energy efficiency.

## 🚀 Technologies Used

- Python
- scikit-learn (for model training)
- Streamlit (for the web app)
- pandas, numpy, Pillow (for data manipulation and display)

## 📁 Project Structure

```
.
├── kangaroo_clean.csv         # Cleaned dataset used for training
├── pred_model.ipynb.ipynb     # Jupyter Notebook for training the model
├── streamlit_app.py           # Streamlit app for predictions
├── kangaroo_clean.sav         # Trained model (generated from notebook)
└── ImmoElizaDeploy.png        # Image used in the app UI
```

## 🔧 How It Works

### 1. Model Training

- Load `kangaroo_clean.csv`
- Encode and scale features
- Train a regression model (e.g. RandomForestRegressor)
- Export model using `pickle` as `kangaroo_clean.sav`

### 2. Web App (Streamlit)

- Load the trained model and dataset
- Display dropdowns and sliders for user input
- Dynamically filter localities based on province
- Format user input and predict property price
- Display the predicted price

## ✅ Features Used

- `type_subtype`
- `province`
- `locality_normalized`
- `kitchenType`
- `bedroomCount`
- `bathroomCount`
- `habitableSurface`
- `gardenSurface`
- `epcScoreMapping`
- `facedeCount`
- `roomCount`
- `version`

## 🛠 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure all files are in the same folder.

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## 📌 Notes

- Ensure column names in user input match those expected by the model (e.g., `epcScoreMapping`).
- The dataset represents real Belgian housing data and can be updated with more recent listings.