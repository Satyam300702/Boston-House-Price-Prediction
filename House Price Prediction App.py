# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 10:18:55 2025

@author: HP
"""

import numpy as np
import os
import pickle
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__),"House_price.sav")
scaler_path = os.path.join(os.path.dirname(__file__), "Boston_Scaler.pkl")
try:
    house_model = pickle.load(open(model_path,"rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
except FileNotFoundError:
    st.error("Model File Not Found")
    st.stop()
    
def House_price_prediction(input_data):
    input_data_as_np_array = np.asarray(input_data).reshape(1,-1)
    scaled_data = scaler.transform(input_data_as_np_array)
    prediction = house_model.predict(scaled_data)
    corrected_prediction = prediction
    return corrected_prediction[0]
    
    
    
    return prediction[0]

def main():
    st.title("Boston House Price Prediction App")
    st.write("### Predict the **median value of homes** using a trained Machine Learning model (in $1000s).")
    
    st.header("Enter House Features Values")
    
    CRIM = st.number_input("CRIM (Per Capita crime rate by town)")
    ZN = st.number_input("ZN (Proportion of residential land zoned for lots over 25,000 sq.ft.)")
    INDUS = st.number_input("INDUS (Proportion of non-retail business acres per town)")
    CHAS = st.number_input("CHAS (Charles River dummy variable)")
    NOX = st.number_input("NOX (Nitric Oxide Concerntration - parts per 10 million)")
    RM = st.number_input("RM (Average number of rooms per dwelling)")
    AGE = st.number_input("AGE (Proportion of owner-occupied units built before 1940)")
    DIS = st.number_input("DIS (Weighted distances to employment centers)")
    RAD = st.number_input("RAD (Index of accessibility to radial highways)")
    TAX = st.number_input("Tax (Full-value property-tax rate per $10,000)")
    PTRATIO = st.number_input("PTRATIO (Pupil-twacher ratio by town)")
    B = st.number_input("B (1000*(Bk - 0.63)^2, where Bk is proportion of blacks by town)")
    LSTAT = st.number_input("LSTAT (% Lower status of the population)")   
    
    model = ""
    
    if st.button("Predict House Price"):
        model = House_price_prediction([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT])
    st.success(model)
    
if __name__ == "__main__":
    main()