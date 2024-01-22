import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Load the pre-trained LSTM model
model_lstm = load_model('D:\model_lstm.h5')

historical_data =  pd.read_csv('HistoricalStockPrices.csv')

# Load transformers and encoders
yeo_johnson_transformer = joblib.load('yeo_johnson_transformer.pkl')
yeo_johnson_target_transformer = joblib.load('yeo_johnson_target_transformer.pkl')
label_encoder_year = joblib.load('label_encoder_year.pkl')
label_encoder_month = joblib.load('label_encoder_month.pkl')
label_encoder_day = joblib.load('label_encoder_day.pkl')

default_date = datetime(2019, 12, 5)

# Streamlit app
st.title("BAT Bangladesh Stock Price Prediction")

# User input for future prediction
date = st.date_input("Select Trading Date:", value=default_date)
open_price = st.number_input("Enter Opening Price of the stock:", value=None, step=0.01)
high_price = st.number_input("Enter Highest Price of that day:", value=None, step=0.01)
low_price = st.number_input("Enter Lowest Price of that day:", value=None, step=0.01)
volume = st.number_input("Enter Traded Volume on that day:", value=None, step=0.00001)

if st.button("Get Predictions"):
    # Extracting additional date-related features
    Month = date.month 
    Day = date.weekday()
    Year = date.year    

    # Apply transformations to user input
    user_input_df = pd.DataFrame(
        [[open_price, high_price, low_price, volume, Month, Day, Year]],
        columns=[' Open', ' High', ' Low', ' Volume', 'Month', 'Day', 'Year']
    )

    # Applying Yeo-Johnson transformation to features
    numeric_columns = [' Open', ' High', ' Low', ' Volume']
    user_input_df[numeric_columns] = yeo_johnson_transformer.transform(user_input_df[numeric_columns])

    # Apply label encoding to categorical features
    user_input_df['Year'] = label_encoder_year.transform(user_input_df['Year'])
    user_input_df['Month'] = label_encoder_month.transform(user_input_df['Month'])
    user_input_df['Day'] = label_encoder_day.transform(user_input_df['Day'])
    
    # Apply Yeo-Johnson transformation to the target variable
    historical_data[[' Close']] = yeo_johnson_target_transformer.transform(historical_data[[' Close']])
     
    # Reshape user input for LSTM model
    user_input_lstm = np.array(user_input_df)
    user_input_lstm = np.reshape(user_input_lstm, (user_input_lstm.shape[0], 1, user_input_lstm.shape[1]))

    # Make predictions using the LSTM model
    prediction = model_lstm.predict(user_input_lstm)
    
    # Apply inverse transformation to get the original scale
    prediction_original = yeo_johnson_target_transformer.inverse_transform(prediction)

    # Display predictions in a styled box
    if date is not None and open_price is not None and high_price is not None and low_price is not None and volume is not None:
        st.success("Predicted Closing Price of the BAT Stock:")
        st.write(f"BDT. {prediction_original[0][0]:,.2f}")
    else:
        st.write("Kindly provide the necessary information for closing price prediction. Best of luck!")

