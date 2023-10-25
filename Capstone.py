import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

r=open("Logistic.pkl","rb")
lr=pickle.load(r)

gender=st.selectbox('select gender',('male','Female'))
SeniorCitizen=st.number_input("Age:",0,100,50)
Partner=st.selectbox('Partnet',('Yes','No'))
Dependents=st.selectbox('Dependents',('Yes','No'))
tenure=st.number_input("tenure:",0,500,1)
PhoneService=st.selectbox('PhoneService',('Yes','No'))
MultipleLines=st.selectbox('MultipleLines',('Yes','No'))
InternetService=st.selectbox('InternetService',('DSL','Fiber optic'))
OnlineSecurity=st.selectbox('OnlineSecurity',('Yes','No'))
OnlineBackup=st.selectbox('OnlineBackup',('Yes','No','No Internet Service'))
DeviceProtection=st.selectbox('DeviceProtection',('Yes','No','No Internet Service'))
TechSupport=st.selectbox('TechSupport',('Yes','No','No Internet Service'))
StreamingTV=st.selectbox('StreamingTV',('Yes','No','No Internet Service'))
StreamingMovies=st.selectbox('StreamingMovies',('Yes','No','No Internet Service'))
Contract=st.selectbox('Contract',('Month-to-month','One year','Two year'))
PaperlessBilling=st.selectbox('PaperlessBilling',('Yes','No'))
PaymentMethod=st.selectbox('PaymentMethod',('Electronic check','Mailed check','Bank transfer','Credit card'))
MonthlyCharges=st.number_input("MonthlyCharges:")
TotalCharges=st.number_input("TotalCharges:")

data=[np.array([gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod])]
data1=[np.array([SeniorCitizen, tenure, MonthlyCharges, TotalCharges])]
#=np.array(data).reshape(1,-1)
prediction=lr.predict(data,data1)
if st.button("Churn Prediction"):
    st.write(str(prediction))
