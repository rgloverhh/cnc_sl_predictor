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

class CNCDepartment:
    def __init__(self, calls, min, sec, fte, nrr):
        self.calls = calls
        self.min = min
        self.sec = sec
        self.fte = fte
        self.nrr = nrr

# items to change on a monthly basis
model_info = "Model updated on 3/3/2025"
sidebar_caption = "Parameters defaulted to department daily averages from February 2025"
updated_end_date = "3/1/2025"
primary_care = CNCDepartment(calls=3114, min=5, sec=48, fte=29.2, nrr=20.4)
cancer_care = CNCDepartment(calls=878, min=5, sec=34, fte=13.5, nrr=23.5)
heart_care = CNCDepartment(calls=1149, min=5, sec=46, fte=17.7, nrr=22.5)
ma_crt = CNCDepartment(calls=598, min=7, sec=52, fte=7.8, nrr=26.6)
ref_phone = CNCDepartment(calls=398, min=4, sec=32, fte=6.0, nrr=32.4)
mychart_nav = CNCDepartment(calls=592, min=2, sec=55, fte=6.9, nrr=19.0)

# standardized text (does not typically need an update)
blended_info = 'The blended models of XGBoost and Linear Regression offer the best accuracy, but may not be suitable for making predictions using parameters far outside of the norm'
linear_info = 'The Linear Regression model is not as accurate as the blended model, but is better for making predictions with more "extreme" parameters'
pcp_timeframes = f"Data Timeframes: 10/3/2022 - {updated_end_date}"
cc_timeframes = f"Data Timeframes: 6/3/2022 - {updated_end_date}"
heart_timeframes = f"Data Timeframes: 1/3/2022 - {updated_end_date}"
ma_timeframes = f"Data Timeframes: 7/3/2023 - {updated_end_date}"
ref_timeframes = f"Data Timeframes: 10/3/2022 - {updated_end_date}"
mychart_nav_timeframes = f"Data Timeframes: 4/7/2024 - {updated_end_date}"
zero_pred = 0.00
hundred_pred = 100.00

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
    st.text("Instructions:\nChoose the department and predictive model below and fill out the parameters on the left.\nCurrent baseline parameters will appear below when you choose a department.")
    st.caption(model_info)
    st.sidebar.header("Input Parameters")
    
    selected_dept = st.radio("Select department:", ["Primary Care", "Cancer Care", "Heart Care", "MA Clinical Resource", "Referrals", "MyChart & Navigation"])
    selected_model = st.radio("Select model:", ["Blended (Linear+XGB)", "Linear Regression"])

    if selected_dept == "Primary Care":
        def_calls = primary_care.calls
        def_aht_min = primary_care.min
        def_aht_sec = primary_care.sec
        def_total_FTEs = primary_care.fte
        def_not_ready = primary_care.nrr
        chosen_lin = pcp_lin_model
        chosen_xgb = pcp_xgb_model
        chosen_alpha = pcp_best_alpha
        sidebar_timeframes = pcp_timeframes
    elif selected_dept == "Cancer Care":
        def_calls = cancer_care.calls
        def_aht_min = cancer_care.min
        def_aht_sec = cancer_care.sec
        def_total_FTEs = cancer_care.fte
        def_not_ready = cancer_care.nrr
        chosen_lin = cc_lin_model
        chosen_xgb = cc_xgb_model
        chosen_alpha = cc_best_alpha
        sidebar_timeframes = cc_timeframes
    elif selected_dept == "Heart Care":
        def_calls = heart_care.calls
        def_aht_min = heart_care.min
        def_aht_sec = heart_care.sec
        def_total_FTEs = heart_care.fte
        def_not_ready = heart_care.nrr
        chosen_lin = heart_lin_model
        chosen_xgb = heart_xgb_model
        chosen_alpha = heart_best_alpha
        sidebar_timeframes = heart_timeframes
    elif selected_dept == "MA Clinical Resource":
        def_calls = ma_crt.calls
        def_aht_min = ma_crt.min
        def_aht_sec = ma_crt.sec
        def_total_FTEs = ma_crt.fte
        def_not_ready = ma_crt.nrr
        chosen_lin = ma_lin_model
        chosen_xgb = ma_xgb_model
        chosen_alpha = ma_best_alpha
        sidebar_timeframes = ma_timeframes
    elif selected_dept == "Referrals":
        def_calls = ref_phone.calls
        def_aht_min = ref_phone.min
        def_aht_sec = ref_phone.sec
        def_total_FTEs = ref_phone.fte
        def_not_ready = ref_phone.nrr
        chosen_lin = ref_lin_model
        chosen_xgb = ref_xgb_model
        chosen_alpha = ref_best_alpha
        sidebar_timeframes = ref_timeframes
    elif selected_dept == "MyChart & Navigation":
        def_calls = mychart_nav.calls
        def_aht_min = mychart_nav.min
        def_aht_sec = mychart_nav.sec
        def_total_FTEs = mychart_nav.fte
        def_not_ready = mychart_nav.nrr
        chosen_lin = mychart_nav_lin_model
        chosen_xgb = mychart_nav_xgb_model
        chosen_alpha = mychart_nav_best_alpha
        sidebar_timeframes = mychart_nav_timeframes

    calls_offered = st.sidebar.number_input(label="Number of Calls", min_value=1, max_value=8000, step=1, value=def_calls)
    aht_minutes = st.sidebar.number_input(label="Average Handle Time (Min)", min_value=1, max_value=10, step=1, value=def_aht_min)
    aht_seconds = st.sidebar.number_input(label="Average Handle Time (Sec)", min_value=0, max_value=59, step=1, value=def_aht_sec)
    total_FTEs = st.sidebar.number_input(label="FTEs Logged In", min_value=1.0, max_value=60.0, step=0.1, value=def_total_FTEs)
    not_ready_rate = st.sidebar.number_input(label="Not Ready Rate (%)", min_value = 1.0, max_value=50.0, step=0.1, value=def_not_ready)
    not_ready_con = not_ready_rate/100
    aht = aht_minutes + (aht_seconds/60)
    st.sidebar.caption(sidebar_caption)
    st.sidebar.caption(sidebar_timeframes)

    if selected_model == "Blended (Linear+XGB)":
        st.caption(blended_info)
        lin_pred = sl_predict(calls_offered, aht, total_FTEs, not_ready_con, chosen_lin)
        xgb_pred = sl_predict(calls_offered, aht, total_FTEs, not_ready_con, chosen_xgb)
        final_pred = blend_predict(lin_pred, xgb_pred, chosen_alpha)
        final_pred *= 100
        with st.container(border=True):
            if final_pred <= 0:
                st.write(f"### 📈 Predicted Service Level: **{zero_pred:.2f}%**")
            elif final_pred >= 100:
                st.write(f"### 📈 Predicted Service Level: **{hundred_pred:.2f}%**")
            else:
                st.write(f"### 📈 Predicted Service Level: **{final_pred:.2f}%**")

    elif selected_model == "Linear Regression":
        st.caption(linear_info)
        lin_pred = sl_predict(calls_offered, aht, total_FTEs, not_ready_con, chosen_lin)
        lin_pred *= 100
        with st.container(border=True):
            if lin_pred <= 0:
                st.write(f"### 📈 Predicted Service Level: **{zero_pred:.2f}%**")
            elif lin_pred >= 100:
                st.write(f"### 📈 Predicted Service Level: **{hundred_pred:.2f}%**")
            else:
                st.write(f"### 📈 Predicted Service Level: **{lin_pred:.2f}%**")

if __name__ == "__main__":
    main()