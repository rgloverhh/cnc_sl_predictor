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
cancer_lin_model = load_model('cancer_lin_model.pkl')
cancer_xgb_model = load_model('cancer_xgb_model.pkl')
cancer_best_alpha = load_model('cancer_best_alpha.pkl')
heart_lin_model = load_model('heart_lin_model.pkl')
heart_xgb_model = load_model('heart_xgb_model.pkl')
heart_best_alpha = load_model('heart_best_alpha.pkl')
ma_lin_model = load_model('ma_lin_model.pkl')
ma_xgb_model = load_model('ma_xgb_model.pkl')
ma_best_alpha = load_model('ma_best_alpha.pkl')
ref_lin_model = load_model('ref_lin_model.pkl')
ref_xgb_model = load_model('ref_xgb_model.pkl')
ref_best_alpha = load_model('ref_best_alpha.pkl')
pain_lin_model = load_model('pain_lin_model.pkl')
pain_xgb_model = load_model('pain_xgb_model.pkl')
pain_best_alpha = load_model('pain_best_alpha.pkl')
neuro_opt_lin_model = load_model('neuro_opt_lin_model.pkl')
neuro_opt_xgb_model = load_model('neuro_opt_xgb_model.pkl')
neuro_opt_best_alpha = load_model('neuro_opt_best_alpha.pkl')

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
model_info = "Model updated on 1/19/2026"
sidebar_caption = "Parameters defaulted to department daily averages from Dec 15 2025 - Jan 16 2026"
updated_end_date = "1/16/2026"

# standardized text (does not typically need an update)
blended_info = 'The blended model is best for making predictions using current state data (i.e. what is currently happening)'
linear_info = 'The linear regression model is best for making predictions for future states (i.e. what happens to the SL when 5 FTEs are added)'
pcp_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
cancer_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
heart_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
ma_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
ref_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
pain_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
neuro_opt_timeframes = f"Data Timeframes: 10/15/2025 - {updated_end_date}"
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
primary_care = CNCDepartment(calls=5616, min=5, sec=56, fte=49.4, nrr=16.7, lin=pcp_lin_model, xg=pcp_xgb_model, alpha=pcp_best_alpha, tf=pcp_timeframes)
cancer_care = CNCDepartment(calls=837, min=5, sec=24, fte=18.0, nrr=15.7, lin=cancer_lin_model, xg=cancer_xgb_model, alpha=cancer_best_alpha, tf=cancer_timeframes)
heart_care = CNCDepartment(calls=1114, min=5, sec=16, fte=16.9, nrr=17.6, lin=heart_lin_model, xg=heart_xgb_model, alpha=heart_best_alpha, tf=heart_timeframes)
ma_crt = CNCDepartment(calls=1032, min=6, sec=11, fte=11.5, nrr=31.5, lin=ma_lin_model, xg=ma_xgb_model, alpha=ma_best_alpha, tf=ma_timeframes)
ref_phone = CNCDepartment(calls=386, min=5, sec=0, fte=6.3, nrr=35.7, lin=ref_lin_model, xg=ref_xgb_model, alpha=ref_best_alpha, tf=ref_timeframes)
pain = CNCDepartment(calls=111, min=4, sec=3, fte=6.8, nrr=18.8, lin=pain_lin_model, xg=pain_xgb_model, alpha=pain_best_alpha, tf=pain_timeframes)
neuro_opt = CNCDepartment(calls=501, min=4, sec=31, fte=10.2, nrr=18.8, lin=neuro_opt_lin_model, xg=neuro_opt_xgb_model, alpha=neuro_opt_best_alpha, tf=neuro_opt_timeframes)

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
    st.text('Instructions:\nChoose the department and predictive model below and fill out the parameters on the left.\nCurrent baseline parameters will appear in the input boxes on the left when you choose a department.\n(Hint: if the Average Handle Time is 5 min and 30 seconds, put the 5 in the "min" box and the 30 in the "sec" box)')
    st.caption(model_info)
    st.sidebar.header("Input Parameters")
    
    selected_dept = st.radio("Select department:", ["Primary Care/Navigation", "Cancer Care", "Heart Care", "MA Clinical Resource", "Referral Calls", "Pain Care", "Neurology/Outpatient Therapy"])
    selected_model = st.radio("Select model:", ["Blended (Linear+XGB)", "Linear Regression"])

    if selected_dept == "Primary Care/Navigation":
        default_values(primary_care)
    elif selected_dept == "Cancer Care":
       default_values(cancer_care)
    elif selected_dept == "Heart Care":
        default_values(heart_care)
    elif selected_dept == "MA Clinical Resource":
        default_values(ma_crt)
    elif selected_dept == "Referral Calls":
        default_values(ref_phone)
    elif selected_dept == "Pain Care":
        default_values(pain)
    elif selected_dept == "Neurology/Outpatient Therapy":
        default_values(neuro_opt)

    calls_offered = st.sidebar.number_input(label="Number of Calls", min_value=1, max_value=10000, step=1, value=def_calls)
    aht_minutes = st.sidebar.number_input(label="Average Handle Time (Min)", min_value=1, max_value=10, step=1, value=def_aht_min)
    aht_seconds = st.sidebar.number_input(label="Average Handle Time (Sec)", min_value=0, max_value=59, step=1, value=def_aht_sec)
    total_FTEs = st.sidebar.number_input(label="FTEs Logged In", min_value=1.0, max_value=100.0, step=1.0, value=def_total_FTEs)
    not_ready_rate = st.sidebar.number_input(label="Not Ready Rate (%)", min_value = 1.0, max_value=50.0, step=1.0, value=def_not_ready)
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