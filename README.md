# Post-traumatic Sepsis Risk Tool

Streamlit interface for the frozen 25-predictor XGBoost model (version 492).

App address: https://trauma-sepsis-risk-app-b5z92ihc6nmcufhrezfsim.streamlit.app/

Run with `pip install -r requirements.txt` and `streamlit run app.py`.

The app uses the original XGBoost booster, training-median imputation and training-derived classification threshold (0.15453055500984192). No retraining or calibration is performed. `model_25.py` contains a losslessly compressed XGBoost JSON model and preprocessing metadata; `inference_25.py` loads it using native XGBoost. No patient identifiers or individual-level training records are included in these new deployment files.

Inputs comprise 25 fixed predictors. Vital signs and laboratory measurements use ICU admission −6 to +6 hours; treatments use 0–6 hours. The endpoint is new sepsis strictly after hour 6 and no later than hour 72 after ICU admission. Blank entries use the saved training medians; all-blank submissions are rejected. "Load example" supplies training-median example values and does not represent a real patient.

Verification: 2,081 nonempty input vectors matched the original Python model exactly; a further all-missing vector verified the empty-form guard. The webpage's inputs and calculation were tested with Streamlit AppTest and a real browser. The old app is preserved on branch `backup-before-25-feature-model-20261110`. Legacy `config.py`, `predict_utils.py`, `input_template.csv` and `trauma_sepsis_xgb_bundle.pkl` are not imported by the new app.

This application provides a research model estimate, not a diagnosis or treatment recommendation. Numerical inputs are processed in the current Streamlit session; this app does not write them to a patient database or file. Do not enter patient identifiers.
