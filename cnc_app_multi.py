import numpy as np
import streamlit as st
import pickle
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# initial function to load the model
def load_model(mdl):
    with open(mdl, 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

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

# initialize default parameter values
def_calls = None
def_aht_min = None
def_aht_sec = None
def_total_FTEs = None
def_not_ready = None
chosen_lin = None
chosen_xgb = None
chosen_alpha = None
sidebar_timeframes = None

# UPDATE items on a monthly basis
model_info = "Model updated on 5/6/2025"
sidebar_caption = "Parameters defaulted to department daily averages from April 2025"
updated_end_date = "5/5/2025"

# standardized text (does not typically need an update)
blended_info = 'The blended model is best for making predictions using current state data (i.e. what is currently happening)'
linear_info = 'The linear regression model is best for making predictions for future states (i.e. what happens to the SL when 5 FTEs are added)'
pcp_timeframes = f"Data Timeframes: 10/3/2022 - {updated_end_date}"
cc_timeframes = f"Data Timeframes: 6/3/2022 - {updated_end_date}"
heart_timeframes = f"Data Timeframes: 1/3/2022 - {updated_end_date}"
ma_timeframes = f"Data Timeframes: 7/3/2023 - {updated_end_date}"
ref_timeframes = f"Data Timeframes: 10/3/2022 - {updated_end_date}"
mychart_nav_timeframes = f"Data Timeframes: 4/7/2024 - {updated_end_date}"
zero_pred = 0.00
hundred_pred = 100.00

# instantiate CNCDepartment Class
class CNCDepartment:
    def __init__(self, calls, min, sec, fte, nrr, lin, xg, alpha, tf):
        self.calls = calls
        self.min = min
        self.sec = sec
        self.fte = fte
        self.nrr = nrr
        self.lin = lin
        self.xg = xg
        self.alpha = alpha
        self.tf = tf

# UPDATE class parameters on a monthly basis
primary_care = CNCDepartment(calls=2767, min=5, sec=43, fte=30.4, nrr=21.9, lin=pcp_lin_model, xg=pcp_xgb_model, alpha=pcp_best_alpha, tf=pcp_timeframes)
cancer_care = CNCDepartment(calls=857, min=5, sec=23, fte=14.1, nrr=20.5, lin=cc_lin_model, xg=cc_xgb_model, alpha=cc_best_alpha, tf=cc_timeframes)
heart_care = CNCDepartment(calls=1111, min=5, sec=24, fte=16.5, nrr=22.6, lin=heart_lin_model, xg=heart_xgb_model, alpha=heart_best_alpha, tf=heart_timeframes)
ma_crt = CNCDepartment(calls=569, min=7, sec=20, fte=9.4, nrr=30.5, lin=ma_lin_model, xg=ma_xgb_model, alpha=ma_best_alpha, tf=ma_timeframes)
ref_phone = CNCDepartment(calls=370, min=4, sec=13, fte=6.7, nrr=41.5, lin=ref_lin_model, xg=ref_xgb_model, alpha=ref_best_alpha, tf=ref_timeframes)
mychart_nav = CNCDepartment(calls=584, min=2, sec=48, fte=5.4, nrr=23.3, lin=mychart_nav_lin_model, xg=mychart_nav_xgb_model, alpha=mychart_nav_best_alpha, tf=mychart_nav_timeframes)

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

def default_values(cnc_class):
    global def_calls, def_aht_min, def_aht_sec, def_total_FTEs, def_not_ready, chosen_lin, chosen_xgb, chosen_alpha, sidebar_timeframes
    def_calls = cnc_class.calls
    def_aht_min = cnc_class.min
    def_aht_sec = cnc_class.sec
    def_total_FTEs =cnc_class.fte
    def_not_ready = cnc_class.nrr
    chosen_lin = cnc_class.lin
    chosen_xgb = cnc_class.xg
    chosen_alpha = cnc_class.alpha
    sidebar_timeframes = cnc_class.tf

# streamlit code
def main():
    st.title("CNC Service Level Predictor")
    st.text("Instructions:\nChoose the department and predictive model below and fill out the parameters on the left.\nCurrent baseline parameters will appear in the input boxes on the left when you choose a department.")
    st.caption(model_info)
    st.sidebar.header("Input Parameters")
    
    selected_dept = st.radio("Select department:", ["Primary Care", "Cancer Care", "Heart Care", "MA Clinical Resource", "Referrals", "MyChart & Navigation"])
    selected_model = st.radio("Select model:", ["Blended (Linear+XGB)", "Linear Regression"])

    if selected_dept == "Primary Care":
        default_values(primary_care)
    elif selected_dept == "Cancer Care":
       default_values(cancer_care)
    elif selected_dept == "Heart Care":
        default_values(heart_care)
    elif selected_dept == "MA Clinical Resource":
        default_values(ma_crt)
    elif selected_dept == "Referrals":
        default_values(ref_phone)
    elif selected_dept == "MyChart & Navigation":
        default_values(mychart_nav)

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