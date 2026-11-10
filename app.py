"""Simple Streamlit interface for the existing 25-predictor model."""
import streamlit as st
from inference_25 import load_model, predict
from model_25 import METADATA

st.set_page_config(page_title='Post-traumatic Sepsis Prediction', page_icon='🩺', layout='wide')
st.markdown("""<style>
.block-container {max-width: 1360px; padding: 2.3rem 2rem 1.2rem;}
header[data-testid="stHeader"], [data-testid="stDecoration"] {display:none;}
footer {display:none;}
h1 {font-family: Arial,sans-serif; font-size: 29px!important; line-height:1.2!important;
    font-weight:700!important; color:#17354c; letter-spacing:-.5px; padding:0!important; margin:0!important;}
.group {font:600 15px Arial,sans-serif; color:#23475f; border-bottom:1px solid #dce5ed;
    padding:0 0 9px; margin:14px 0 7px;}
div[data-testid="stVerticalBlock"] {gap:.45rem;}
div[data-testid="stNumberInput"] label p, div[data-testid="stSelectbox"] label p {font-size:12px!important; color:#354d60;}
div[data-baseweb="input"],div[data-baseweb="select"]>div {background:#f7f9fc; border-color:#d7e1ea; border-radius:5px;}
div[data-testid="stNumberInput"] button {display:none;}
div[data-testid="stNumberInput"] input {font-size:14px;}
button[kind="primary"] {background:#0875bd; border:0;}
div[data-testid="stButton"] button {min-height:39px;}
.result {display:flex; align-items:center; justify-content:space-between; gap:24px;
    background:#f0f7fc; border:1px solid #d8e7f2; border-radius:7px; padding:15px 25px;
    margin-top:3px; color:#17354c;}
.result-label {font:600 17px Arial,sans-serif;}
.risk {font:700 43px Arial,sans-serif; color:#0766a8; line-height:1.1; letter-spacing:-1px; white-space:nowrap;}
.risk span {font-size:25px; padding-left:2px;}
.result-note {font:12px Arial,sans-serif; color:#6a7f8f; margin-top:5px;}
.footer {font:11px Arial,sans-serif; color:#82909d; margin-top:14px;}
@media(max-width:640px) {.block-container{padding:1.5rem 1rem 1rem;}h1{font-size:24px!important;}.result{padding:14px;}}
</style>""", unsafe_allow_html=True)

GROUPS=[
('Demographics',[('age','Age (years)',18.,120.),('sex','Sex',0.,1.,'sex')]),
('Treatments in the first 6 hours',[('invasive_vent_0_6h','Invasive mechanical ventilation',0.,1.,'binary'),('any_vasopressor_0_6h','Any vasopressor',0.,1.,'binary')]),
('Vital signs & physiology',[
('heart_rate_min_m6_p6h','Heart rate · minimum (beats/min)',0.,350.),('heart_rate_mean_m6_p6h','Heart rate · mean (beats/min)',0.,350.),('sbp_max_m6_p6h','Systolic BP · maximum (mmHg)',0.,400.),('dbp_mean_m6_p6h','Diastolic BP · mean (mmHg)',0.,250.),('map_min_m6_p6h','MAP · minimum (mmHg)',0.,300.),('map_mean_m6_p6h','MAP · mean (mmHg)',0.,300.),('respiratory_rate_mean_m6_p6h','Respiratory rate · mean (/min)',0.,100.),('respiratory_rate_max_m6_p6h','Respiratory rate · maximum (/min)',0.,100.),('spo2_min_m6_p6h','SpO₂ · minimum (%)',0.,100.),('spo2_mean_m6_p6h','SpO₂ · mean (%)',0.,100.),('spo2_max_m6_p6h','SpO₂ · maximum (%)',0.,100.),('gcs_min_m6_p6h','GCS · minimum (points)',3.,15.),('shock_index_m6_p6h','Shock index (ratio)',0.,10.)]),
('Laboratory results',[('wbc','White blood cell count (10⁹/L)',0.,1000.),('platelet','Platelet count (10⁹/L)',0.,3000.),('rdw','Red cell distribution width (%)',0.,100.),('chloride','Chloride (mEq/L)',0.,250.),('glucose','Glucose (mg/dL)',0.,3000.),('calcium_total','Total calcium (mg/dL)',0.,40.),('inr','INR',0.,30.),('ptt','aPTT (s)',0.,500.)])]
assert set(METADATA['features'])=={e[0] for _,es in GROUPS for e in es}

@st.cache_resource
def frozen_model(): return load_model()

def clear_result():
    st.session_state.pop('prediction',None)
    st.session_state['is_example']=False

def load_example():
    for f,v in zip(METADATA['features'],METADATA['medians']):
        st.session_state[f]=(['Female','Male'][int(v)] if f=='sex' else ['No','Yes'][int(v)]) if f in ['sex','invasive_vent_0_6h','any_vasopressor_0_6h'] else round(float(v),2)
    st.session_state.pop('prediction',None)
    st.session_state['is_example']=True

def reset_inputs():
    for f in METADATA['features']: st.session_state[f]=None
    clear_result()

def render_fields(entries,ncols):
    cols=st.columns(ncols)
    for i,e in enumerate(entries):
        f,label,lo,hi,*kind=e
        with cols[i%ncols]:
            if kind:
                labels={0:'Female',1:'Male'} if kind[0]=='sex' else {0:'No',1:'Yes'}
                st.selectbox(label,list(labels.values()),index=None,placeholder='—',key=f,on_change=clear_result)
            else:
                st.number_input(label,min_value=lo,max_value=hi,value=None,step=1. if f in ['age','gcs_min_m6_p6h'] else .01,format='%.0f' if f in ['age','gcs_min_m6_p6h'] else '%.2f',placeholder='—',key=f,on_change=clear_result)


title, example, clear = st.columns([5.2,1.0,.7], gap='medium')
with title:
    st.title('Post-traumatic Sepsis Prediction')
example.button('Load example', on_click=load_example, use_container_width=True)
clear.button('Clear', on_click=reset_inputs, use_container_width=True)

st.markdown('<div class="group">Patient information & early treatment</div>', unsafe_allow_html=True)
render_fields(GROUPS[0][1] + GROUPS[1][1], 4)
for name, entries in GROUPS[2:]:
    st.markdown('<div class="group">' + name + '</div>', unsafe_allow_html=True)
    render_fields(entries, 4)

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
action, hint = st.columns([1.05, 4.5], gap='medium')
submitted = action.button('Calculate risk', type='primary', use_container_width=True)
hint.caption('Leave unavailable values blank; missing values are filled automatically.')
if submitted:
    try:
        vals = {f:st.session_state.get(f) for f in METADATA['features']}
        for f in ['sex','invasive_vent_0_6h','any_vasopressor_0_6h']:
            vals[f] = {'Female':0,'Male':1,'No':0,'Yes':1,None:None}[vals[f]]
        for a,b in [('heart_rate_min_m6_p6h','heart_rate_mean_m6_p6h'),('map_min_m6_p6h','map_mean_m6_p6h'),('respiratory_rate_mean_m6_p6h','respiratory_rate_max_m6_p6h'),('spo2_min_m6_p6h','spo2_mean_m6_p6h'),('spo2_mean_m6_p6h','spo2_max_m6_p6h')]:
            if vals[a] is not None and vals[b] is not None and vals[a] > vals[b]:
                raise ValueError('Check the minimum, mean and maximum values: their order is inconsistent.')
        if vals['gcs_min_m6_p6h'] is not None and not float(vals['gcs_min_m6_p6h']).is_integer():
            raise ValueError('GCS must be an integer.')
        st.session_state['prediction'] = predict(frozen_model(), [vals[f] for f in METADATA['features']])
    except ValueError as exc:
        st.session_state.pop('prediction', None)
        st.error(str(exc))

result = st.session_state.get('prediction')
note = ''
if result:
    risk = f'{result["probability"] * 100:.1f}<span>%</span>'
    if st.session_state.get('is_example'):
        note = '<div class="result-note">Example inputs</div>'
    elif result['imputed']:
        note = f'<div class="result-note">{result["imputed"]} missing inputs filled</div>'
else:
    risk = '—'
st.markdown('<div class="result"><div><div class="result-label">Estimated sepsis risk</div>' + note + '</div><div class="risk">' + risk + '</div></div>', unsafe_allow_html=True)
st.markdown('<div class="footer">For research use only.</div>', unsafe_allow_html=True)
