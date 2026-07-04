import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_numeric_dtype

MODEL_PATH = 'car_price_model.joblib'
CSV_PATH = 'car_price_prediction_.csv'


def build_and_train(df: pd.DataFrame):
    if 'Price' not in df.columns:
        raise ValueError("Dataset must contain a 'Price' column")
    X = df.drop('Price', axis=1)
    y = df['Price']

    categorical_cols = [c for c in X.columns if X[c].dtype == 'object' or not is_numeric_dtype(X[c])]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, X


st.title('Car Price Prediction')

if not os.path.exists(CSV_PATH):
    st.error(f"Dataset not found: {CSV_PATH}. Put the CSV in the workspace.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Load or train model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f'Failed to load existing model: {e}')

if model is None:
    if st.button('Train model'):
        with st.spinner('Training model — this may take a while'):
            model, _ = build_and_train(df)
        st.success('Model trained and saved')
    else:
        st.info('No trained model found. Click "Train model" to train from the CSV.')
        st.stop()

# Build input form
X_sample = df.drop(columns=['Price'])
categorical_cols = [c for c in X_sample.columns if X_sample[c].dtype == 'object' or not is_numeric_dtype(X_sample[c])]
numeric_cols = [c for c in X_sample.columns if c not in categorical_cols]

st.header('Input features')
input_data = {}
with st.form('input_form'):
    for col in numeric_cols:
        median = float(X_sample[col].median()) if not X_sample[col].isna().all() else 0.0
        val = st.number_input(col, value=median)
        input_data[col] = val

    for col in categorical_cols:
        options = X_sample[col].dropna().unique().tolist()
        if 1 <= len(options) <= 100:
            val = st.selectbox(col, options)
        else:
            val = st.text_input(col, value='')
        input_data[col] = val

    submitted = st.form_submit_button('Predict')

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)
        price = float(pred[0])
        st.metric('Predicted Price', f'${price:,.2f}')
    except Exception as e:
        st.error(f'Prediction failed: {e}')
import streamlit as st
import pandas as pd


st.title("Car Price Prediction")


# Load the dataset
df = pd.read_csv('car_price_prediction_.csv')
st.write("Dataset preview:")
st.dataframe(df.head())
st.write("Dataset description:")
st.write(df.describe())
st.write("Dataset info:")
st.write(df.info())







