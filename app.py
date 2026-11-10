"""Streamlit interface for the frozen 25-predictor trauma sepsis model."""
import streamlit as st
from inference_25 import load_model, predict
from model_25 import METADATA

st.set_page_config(page_title='Post-traumatic Sepsis Risk Tool',page_icon='🩺',layout='wide')
st.markdown('''<style>
.block-container{max-width:1400px;padding:2rem 2rem 1rem}header[data-testid="stHeader"]{background:transparent}footer{display:none}
h1{font-family:Arial,sans-serif;font-size:30px!important;color:#172e43;letter-spacing:-.5px;padding:0!important;margin-bottom:5px!important}
.subtitle{font:14px Arial;color:#607383;margin-bottom:20px}.context{display:flex;justify-content:space-between;gap:20px;background:#f3f8fc;border:1px solid #dce7ef;border-radius:8px;padding:15px 20px;margin-bottom:24px;color:#29465d;font:14px Arial}.context small{display:block;color:#718695;font-size:10px;letter-spacing:1px;margin-bottom:5px}
.group{font:700 11px Arial;letter-spacing:1px;text-transform:uppercase;color:#4d667a;border-bottom:1px solid #dce7ef;padding:5px 0 8px;margin:14px 0 7px}
div[data-testid="stVerticalBlock"]{gap:.55rem}div[data-testid="stNumberInput"] label p,div[data-testid="stSelectbox"] label p{font-size:12px!important}div[data-baseweb="input"],div[data-baseweb="select"]>div{background:#f5f9fc;border-color:#dce7ef;border-radius:5px}div[data-testid="stNumberInput"] button{display:none}div[data-testid="stNumberInput"] input{font-size:14px}
.result-card{background:linear-gradient(140deg,#edf6fc,#fff);border:1px solid #d8e7f2;border-radius:10px;padding:23px;color:#172e43}.result-card h3{font:700 19px Arial;margin:0 0 24px}.eyebrow{color:#607383;font:12px Arial}.risk{font:700 76px Arial;color:#075c9e;letter-spacing:-3px;margin:7px 0}.risk span{font-size:30px;letter-spacing:0}.meter{height:7px;background:#e2edf5;border-radius:6px;overflow:hidden;margin:16px 0}.meter i{display:block;height:100%;background:#0875c9}.classification{font:600 13px Arial;color:#075c9e;line-height:1.5}.threshold{font:11px Arial;color:#607383;line-height:1.6}.counts{display:flex;gap:55px;border-top:1px solid #dce7ef;padding-top:16px;margin-top:18px}.counts small{color:#607383;font:11px Arial;display:block}.counts strong{display:block;font:700 20px Arial;margin-top:6px}.example{font:11px Arial;color:#718695;line-height:1.5;margin-top:18px}.notes{color:#607383;font:12px Arial;line-height:1.6;padding:2px 4px}.notes h4{font:600 13px Arial;color:#36546b;margin:18px 0 5px}.bottom{border-top:1px solid #dce7ef;color:#8293a0;font:10px Arial;padding-top:12px;margin-top:22px}
button[kind="primary"]{background:#0875c9;border:0}div[data-testid="stForm"]{border:0;padding:0}
</style>''',unsafe_allow_html=True)

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
                st.selectbox(label,list(labels.values()),index=None,placeholder='Not available',key=f,on_change=clear_result)
            else:
                st.number_input(label,min_value=lo,max_value=hi,value=None,step=1. if f in ['age','gcs_min_m6_p6h'] else .01,format='%.0f' if f in ['age','gcs_min_m6_p6h'] else '%.2f',placeholder='Not available',key=f,on_change=clear_result)

st.title('Post-traumatic Sepsis Risk Tool')
st.markdown('<div class="subtitle">XGBoost prediction model · 25 predictors · Streamlit</div>',unsafe_allow_html=True)
st.markdown('<div class="context"><div><small>POPULATION</small>Adults with severe trauma</div><div><small>PREDICTION TIME</small>6 hours after ICU admission</div><div><small>OUTCOME WINDOW</small>New sepsis: &gt;6 to ≤72 hours</div></div>',unsafe_allow_html=True)
left,right=st.columns([3,1],gap='large')
with left:
    heading,example,clear=st.columns([5,1.2,.8])
    heading.markdown('#### Patient information')
    example.button('Load example',on_click=load_example,use_container_width=True)
    clear.button('Clear',on_click=reset_inputs,use_container_width=True)
    st.caption('Use the indicated units. Leave unavailable measurements blank.')
    st.markdown('<div class="group">Demographics & treatments in the first 6 hours</div>',unsafe_allow_html=True)
    render_fields(GROUPS[0][1]+GROUPS[1][1],4)
    for name,entries in GROUPS[2:]:
        st.markdown('<div class="group">'+name+'</div>',unsafe_allow_html=True);render_fields(entries,4)
    calc,count=st.columns([1,3])
    submitted=calc.button('Calculate risk',type='primary',use_container_width=True)
    observed=sum(st.session_state.get(f) is not None for f in METADATA['features'])
    count.caption(f'{observed} of 25 values entered')
    if submitted:
        try:
            vals={f:st.session_state.get(f) for f in METADATA['features']}
            for f in ['sex','invasive_vent_0_6h','any_vasopressor_0_6h']:
                vals[f]={'Female':0,'Male':1,'No':0,'Yes':1,None:None}[vals[f]]
            for a,b in [('heart_rate_min_m6_p6h','heart_rate_mean_m6_p6h'),('map_min_m6_p6h','map_mean_m6_p6h'),('respiratory_rate_mean_m6_p6h','respiratory_rate_max_m6_p6h'),('spo2_min_m6_p6h','spo2_mean_m6_p6h'),('spo2_mean_m6_p6h','spo2_max_m6_p6h')]:
                if vals[a] is not None and vals[b] is not None and vals[a]>vals[b]: raise ValueError('Check the minimum, mean and maximum values: their order is inconsistent.')
            if vals['gcs_min_m6_p6h'] is not None and not float(vals['gcs_min_m6_p6h']).is_integer(): raise ValueError('GCS must be an integer.')
            st.session_state['prediction']=predict(frozen_model(),[vals[f] for f in METADATA['features']])
        except ValueError as exc:
            st.session_state.pop('prediction',None); st.error(str(exc))
with right:
    r=st.session_state.get('prediction')
    if r:
        pct=r['probability']*100
        example_text='<div class="example">Training-median example; not a real patient.</div>' if st.session_state.get('is_example') else ''
        content=f'<div class="eyebrow">Predicted probability of sepsis</div><div class="risk">{pct:.1f}<span>%</span></div><div class="meter"><i style="width:{pct:.4f}%"></i></div><div class="classification">{"Above" if r["positive"] else "Below"} the model classification threshold</div><p class="threshold">Training-derived classification threshold: {METADATA["threshold"]*100:.2f}%. This is not a clinical intervention threshold.</p><div class="counts"><div><small>Observed inputs</small><strong>{r["observed"]} / 25</strong></div><div><small>Imputed inputs</small><strong>{r["imputed"]}</strong></div></div>{example_text}'
    else: content='<p class="eyebrow">Enter the available information and select Calculate risk.</p><div class="risk">—</div>'
    st.markdown('<div class="result-card"><h3>Prediction result</h3>'+content+'</div>',unsafe_allow_html=True)
    st.markdown('<div class="notes"><h4>Measurement windows</h4>Vital signs and laboratory results: −6 to +6 hours relative to ICU admission. Treatments: 0–6 hours after admission.<h4>Missing measurements</h4>Blank fields use the medians saved with the trained model.<h4>Interpretation</h4>A model estimate for research use, not a diagnosis or treatment recommendation.<h4>Privacy</h4>Enter numerical measurements only; no patient identifiers. Inputs are processed for the current session and are not written to a patient database or file by this app.</div>',unsafe_allow_html=True)
st.markdown('<div class="bottom">Frozen XGBoost · 25 predictors · Version 492 · Streamlit</div>',unsafe_allow_html=True)
