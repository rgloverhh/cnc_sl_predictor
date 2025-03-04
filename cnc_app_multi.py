import numpy as np
import streamlit as st
import pickle
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

#inputs - calls offered, AHT, not ready rate, total ftes, fte callouts
def load_model(mdl):
    with open(mdl, 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

# items to change on a monthly basis
model_info = "Model updated on March 3 2025 - default parameters are the daily averages from February 2024"

pcp_baselines = "Primary Care Baseline Parameters\nCalls: 3114\nAHT: 5 min, 48 sec\nTotal_FTEs: 29.2\nNot Ready Rate: 20.4%"

pcp_timeframes = "Data Timeframes: 10/3/2022 - 3/1/2025"
cc_timeframes = "Data Timeframes: 6/3/2022 - 3/1/2025"
heart_timeframes = "Data Timeframes: 1/3/2022 - 3/1/2025"
ma_timeframes = "Data Timeframes: 7/3/2023 - 3/1/2025"
ref_timeframes = "Data Timeframes: 10/3/2022 - 3/1/2025"
mychart_nav_timeframes = "Data Timeframes: 4/7/2024 - 3/1/2025"


# standardized text
blended_info = 'Recommended: This model offers the best accuracy, but may not be suitable for making predictions using parameters far outside of the norm'
linear_info = 'This model is not as accurate as the blended model, but is better for making predictions with more "extreme" parameters'
zero_pred = 0.00
hundred_pred = 1.00

# predictive models
pcp_lin_model = load_model('pcp_lin_model.pkl')
pcp_xgb_model = load_model('pcp_xgb_model.pkl')
pcp_best_alpha = load_model('pcp_best_alpha.pkl')
cc_lin_model = load_model('cc_lin_model.pkl')
cc_xgb_model = load_model('cc_xgb_model.pkl')
cc_best_alpha = load_model('cc_best_alpha.pkl')
heart_lin_model = load_model('heart_lin_model.pkl')
heart_xgb_model = load_model('heart_xgb_model.pkl')
heart_best_alpha = load_model('heart_best_alpha.pkl')
ma_lin_model = load_model('ma_lin_model.pkl')
ma_xgb_model = load_model('ma_xgb_model.pkl')
ma_best_alpha = load_model('ma_best_alpha.pkl')
ref_lin_model = load_model('ref_lin_model.pkl')
ref_xgb_model = load_model('ref_xgb_model.pkl')
ref_best_alpha = load_model('ref_best_alpha.pkl')
mychart_nav_lin_model = load_model('mychart_nav_lin_model.pkl')
mychart_nav_xgb_model = load_model('mychart_nav_xgb_model.pkl')
mychart_nav_best_alpha = load_model('mychart_nav_best_alpha.pkl')

# functions
def preprocess_input(calls, avg_handle_time, total_FTEs, not_ready_rate):
    log_calls = np.log1p(calls)
    calls_per_FTE = calls/total_FTEs
    calls_x_AHT = calls * avg_handle_time
    return np.array([[calls, log_calls, avg_handle_time, calls_per_FTE, calls_x_AHT, total_FTEs, not_ready_rate]])

def sl_predict(calls, avg_handle_time, total_FTEs, not_ready_rate, model):
    input_data = preprocess_input(calls, avg_handle_time, total_FTEs, not_ready_rate)
    pred_sl = model.predict(input_data)[0]
    return pred_sl

def blend_predict(lin_pred, xgb_pred, best_alpha):
    final_pred = (best_alpha * xgb_pred) + ((1 - best_alpha) * lin_pred)
    return final_pred

# streamlit code
def main():
    st.title("CNC Service Level Predictor")
    st.text("Instructions:\nChoose the department and predictive model below and\nfill out the parameters on the left.\nCurrent baseline parameters will appear below when you choose a department.")
    st.caption(model_info)
    st.sidebar.header("Input Parameters")
    
    calls_offered = st.sidebar.number_input(label="Number of Calls", min_value=1, max_value=8000, step=1, value=1000)
    aht_minutes = st.sidebar.number_input(label="Average Handle Time (Min)", min_value=1, max_value=10, step=1, value=5)
    aht_seconds = st.sidebar.number_input(label="Average Handle Time (Sec)", min_value=0, max_value=59, step=1, value=30)
    total_FTEs = st.sidebar.number_input(label="FTEs Logged In", min_value=1.0, max_value=60.0, step=0.1, value=15.0)
    not_ready_rate = st.sidebar.number_input(label="Not Ready Rate (%)", min_value = 1.0, max_value=50.0, step=0.1, value=20.0)
    not_ready_con = not_ready_rate/100
    aht = aht_minutes + (aht_seconds/60)

    selected_dept = st.radio("Select department:", ["Primary Care", "Cancer Care", "Heart Care", "MA CRT Team", "Referral Calls"])
    selected_model = st.radio("Select model:", ["Blended (Linear+XGB)", "Linear Regression"])


    if selected_dept == "Primary Care" and selected_model == "Blended (Linear+XGB)":
        st.text(pcp_baselines)
        st.text(blended_info)
        lin_pred = sl_predict(calls_offered, aht, total_FTEs, not_ready_con, pcp_lin_model)
        xgb_pred = sl_predict(calls_offered, aht, total_FTEs, not_ready_con, pcp_xgb_model)
        final_pred = blend_predict(lin_pred, xgb_pred, pcp_best_alpha)
        
        with st.container(border=True):
            if final_pred <= 0:
                st.write(f"### 📈 Predicted Service Level: **{zero_pred:.2f}%**")
            elif final_pred >= 1:
                st.write(f"### 📈 Predicted Service Level: **{hundred_pred:.2f}%**")
            else:
                st.write(f"### 📈 Predicted Service Level: **{final_pred:.2f}%**")
        st.sidebar.caption(pcp_timeframes)

    if selected_dept == "Primary Care" and selected_model == "Linear Regression":
        st.text(pcp_baselines)
        st.text(linear_info)
        lin_pred = sl_predict(calls_offered, aht, total_FTEs, not_ready_con, pcp_lin_model)
        
        with st.container(border=True):
            if lin_pred <= 0:
                st.write(f"### 📈 Predicted Service Level: **{zero_pred:.2f}%**")
            elif lin_pred >= 1:
                st.write(f"### 📈 Predicted Service Level: **{hundred_pred:.2f}%**")
            else:
                st.write(f"### 📈 Predicted Service Level: **{final_pred:.2f}%**")
        st.sidebar.caption(pcp_timeframes)


if __name__ == "__main__":
    main()