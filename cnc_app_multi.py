import numpy as np
import streamlit as st
import pickle
from xgboost import XGBRegressor

#inputs - calls offered, AHT, not ready rate, total ftes, fte callouts

def load_model(mdl):
    with open(mdl, 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

model_pcp = load_model('pcp_xg_model.pkl')
model_cc = load_model('cc_xg_model.pkl')
model_hc = load_model('heart_xg_model.pkl')
model_ma = load_model('ma_xg_model.pkl')
model_ref = load_model('ref_xg_model.pkl')

def predict_pcp(input1, input2, input3, input4):
    sl = model_pcp.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_cc(input1, input2, input3, input4):
    sl = model_cc.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_hc(input1, input2, input3, input4):
    sl = model_hc.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_ma(input1, input2, input3, input4):
    sl = model_ma.predict([[input1, input2, input3, input4]])
    return sl[0]

def predict_ref(input1, input2, input3, input4):
    sl = model_ref.predict([[input1, input2, input3, input4]])
    return sl[0]


def main():
    st.title("CNC Service Level Predictor")
    st.text("Please fill in the responses below to predict service level")
    st.caption("Model updated Jan 8 2024 - default values are the daily averages from Dec 2024")
    st.sidebar.header("CNC Call Departments")
    selected_model = st.sidebar.radio("Select department:", ["Primary Care", "Cancer Care", "Heart Care", "MA CRT Team", "Referral Calls"])

    if selected_model == "Primary Care":
        with st.container(border=True):
            st.text("1. Call Volumes")
            calls_offered = st.number_input(label="Enter a call volume between 500 and 4000", min_value=500, max_value=4000, step=1, value=2731)
        with st.container(border=True):
            st.text("2. Average Handle Time")
            aht_minutes = st.number_input(label="Enter AHT minutes between 4 and 6", min_value=4, max_value=6, step=1, value=5)
            aht_seconds = st.number_input(label="Enter AHT seconds between 0 and 59", min_value=0, max_value=59, step=1, value=30)
        with st.container(border=True):
            st.text("3. Not Ready Rate")
            not_ready = st.number_input(label="Enter Not Ready Rate between 15 and 35", min_value=15.0, max_value=35.0, step=0.1, value=18.8)
        with st.container(border=True):
            st.text("4. FTEs Logged In (Use Power BI CNC Call Metrics Staffing as a guide)")
            ftes_logged_in = st.number_input(label="Enter FTEs between 20 and 45", min_value=20.0, max_value=45.0, step=0.1, value=32.9)
        not_ready_con = not_ready/100
        aht = aht_minutes + (aht_seconds/60)
        sl_prediction_temp = predict_pcp(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        with st.container(border=True):
            st.header("Primary Care Service Level Prediction")
            if sl_prediction <= 0:
                st.subheader("0%")
            elif sl_prediction >= 100:
                st.subheader("100%")
            else:
                st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)")
        st.sidebar.caption("Data timeframes: 10/3/2022-1/7/2025")
        st.sidebar.caption("Current accuracy: 91.94% (3.54%)")


    elif selected_model == "Cancer Care":
        with st.container(border=True):
            st.text("1. Call Volumes")
            calls_offered = st.number_input(label="Enter a call volume between 300 and 1400", min_value=300, max_value=1400, step=1, value=817)
        with st.container(border=True):
            st.text("2. Average Handle Time")
            aht_minutes = st.number_input(label="Enter AHT minutes between 5 and 7", min_value=5, max_value=7, step=1, value=5)
            aht_seconds = st.number_input(label="Enter AHT seconds between 0 and 59", min_value=0, max_value=59, step=1, value=20)
        with st.container(border=True):
            st.text("3. Not Ready Rate")
            not_ready = st.number_input(label="Enter Not Ready Rate between 15 and 35", min_value=15.0, max_value=35.0, step=0.1, value=22.9)
        with st.container(border=True):
            st.text("4. FTEs Logged In (Use Power BI CNC Call Metrics Staffing as a guide)")
            ftes_logged_in = st.number_input(label="Enter FTEs between 7 and 17", min_value=7.0, max_value=17.0, step=0.1, value=13.6)
        not_ready_con = not_ready/100
        aht = aht_minutes + (aht_seconds/60)
        sl_prediction_temp = predict_cc(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        with st.container(border=True):
            st.header("Cancer Care Service Level Prediction")
            if sl_prediction <= 0:
                st.subheader("0%")
            elif sl_prediction >= 100:
                st.subheader("100%")
            else:
                st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 6/3/2022-1/7/2025")
        st.sidebar.caption("Current accuracy: 84.82% (4.00%)")


    elif selected_model == "Heart Care":
        with st.container(border=True):
            st.text("1. Call Volumes")
            calls_offered = st.number_input(label="Enter a call volume between 300 and 1400", min_value=300, max_value=1400, step=1, value=830)
        with st.container(border=True):
            st.text("2. Average Handle Time")
            aht_minutes = st.number_input(label="Enter AHT minutes between 4 and 7", min_value=4, max_value=7, step=1, value=5)
            aht_seconds = st.number_input(label="Enter AHT seconds between 0 and 59", min_value=0, max_value=59, step=1, value=12)
        with st.container(border=True):
            st.text("3. Not Ready Rate")
            not_ready = st.number_input(label="Enter Not Ready Rate between 15 and 35", min_value=15.0, max_value=35.0, step=0.1, value=20.0)
        with st.container(border=True):
            st.text("4. FTEs Logged In (Use Power BI CNC Call Metrics Staffing as a guide)")
            ftes_logged_in = st.number_input(label="Enter FTEs between 7 and 18", min_value=7.0, max_value=18.0, step=0.1, value=12.8)
        not_ready_con = not_ready/100
        aht = aht_minutes + (aht_seconds/60)
        sl_prediction_temp = predict_hc(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        with st.container(border=True):
            st.header("Heart Care Service Level Prediction")
            if sl_prediction <= 0:
                st.subheader("0%")
            elif sl_prediction >= 100:
                st.subheader("100%")
            else:
                st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 1/3/2022-1/7/2025")
        st.sidebar.caption("Current accuracy: 92.15% (2.33%)")


    elif selected_model == "MA CRT Team":
        with st.container(border=True):
            st.text("1. Call Volumes")
            calls_offered = st.number_input(label="Enter a call volume between 200 and 1100", min_value=200, max_value=1100, step=1, value=615)
        with st.container(border=True):
            st.text("2. Average Handle Time")
            aht_minutes = st.number_input(label="Enter AHT minutes between 5 and 9", min_value=5, max_value=9, step=1, value=7)
            aht_seconds = st.number_input(label="Enter AHT seconds between 0 and 59", min_value=0, max_value=59, step=1, value=12)
        with st.container(border=True):
            st.text("3. Not Ready Rate")
            not_ready = st.number_input(label="Enter Not Ready Rate between 20 and 45", min_value=20.0, max_value=45.0, step=0.1, value=34.4)
        with st.container(border=True):
            st.text("4. FTEs Logged In (Use Power BI CNC Call Metrics Staffing as a guide)")
            ftes_logged_in = st.number_input(label="Enter FTEs between 4 and 15", min_value=4.0, max_value=15.0, step=0.1, value=8.3)
        not_ready_con = not_ready/100
        aht = aht_minutes + (aht_seconds/60)
        sl_prediction_temp = predict_ma(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        with st.container(border=True):
            st.header("MA CRT Service Level Prediction")
            if sl_prediction <= 0:
                st.subheader("0%")
            elif sl_prediction >= 100:
                st.subheader("100%")
            else:
                st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 7/3/2023-1/7/2025")
        st.sidebar.caption("Current accuracy: 84.34% (9.43%)")


    elif selected_model == "Referral Calls":
        with st.container(border=True):
            st.text("1. Call Volumes")
            calls_offered = st.number_input(label="Enter a call volume between 150 and 600", min_value=150, max_value=600, step=1, value=330)
        with st.container(border=True):
            st.text("2. Average Handle Time")
            aht_minutes = st.number_input(label="Enter AHT minutes between 4 and 7", min_value=4, max_value=7, step=1, value=4)
            aht_seconds = st.number_input(label="Enter AHT seconds between 0 and 59", min_value=0, max_value=59, step=1, value=26)
        with st.container(border=True):
            st.text("3. Not Ready Rate")
            not_ready = st.number_input(label="Enter Not Ready Rate between 10 and 30", min_value=10.0, max_value=30.0, step=0.1, value=26.5)
        with st.container(border=True):
            st.text("4. FTEs Logged In (Use Power BI CNC Call Metrics Staffing as a guide)")
            ftes_logged_in = st.number_input(label="Enter FTEs between 2 and 8", min_value=2.0, max_value=8.0, step=0.1, value=5.5)
        not_ready_con = not_ready/100
        aht = aht_minutes + (aht_seconds/60)
        sl_prediction_temp = predict_ref(calls_offered, aht, not_ready_con, ftes_logged_in)
        sl_prediction = sl_prediction_temp*100
        with st.container(border=True):
            st.header("Referral Calls Service Level Prediction")
            if sl_prediction <= 0:
                st.subheader("0%")
            elif sl_prediction >= 100:
                st.subheader("100%")
            else:
                st.subheader(f"{sl_prediction}%")
        st.sidebar.caption("Model: eXtreme Gradient Boosting (XGBoost)") 
        st.sidebar.caption("Data timeframes: 10/3/2022-1/7/2025")
        st.sidebar.caption("Current accuracy: 84.42% (4.37%)")        
   
if __name__ == "__main__":
    main()