from pathlib import Path

softwareName = "创伤后脓毒症早期风险预测系统"
softwareVersion = "V1.0"

appTitle = f"{softwareName} {softwareVersion}"
resultTitle = "预测结果"

systemDesc = "用于创伤后脓毒症早期风险预测。"
moduleDesc = "支持单例预测和批量预测。"
footerText = "结果仅供研究参考。"

baseDir = Path(__file__).resolve().parent

bundlePath = baseDir / "trauma_sepsis_xgb_bundle.pkl"
templatePath = baseDir / "input_template.csv"
batchResultPath = baseDir / "batch_prediction_results.csv"
previewResultPath = baseDir / "train_preview_results.csv"

# 仅本地训练时使用；部署时不会调用这个路径
trainDataPath = Path(
    r"C:\Users\qcx\Desktop\新建文件夹 (8)\remove_high_impact_analysis\dataset_after_removing_high_impact_patients - 副本.csv"
)

labelCol = "target_label"

selectedFeatures = [
    "lactate_n_meas",
    "pao2_worst",
    "mech_vent_6h",
    "ph_worst",
    "ph_n_meas",
    "spo2_h6",
    "bmi",
    "paco2_worst",
    "potassium_n_meas",
    "chloride_worst"
]

riskThreshold = 0.5

xgbParams = {
    "n_estimators": 199,
    "max_depth": 5,
    "learning_rate": 0.027326575518718164,
    "subsample": 0.7095566117240375,
    "colsample_bytree": 0.7924850334907154,
    "min_child_weight": 19,
    "gamma": 2.530055539137471,
    "reg_alpha": 4.603232560980015,
    "reg_lambda": 4.6370770359317355,
    "scale_pos_weight": 1.64794994824111,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1
}

featureNameMap = {
    "lactate_n_meas": "Lactate measurement count",
    "pao2_worst": "Worst PaO2",
    "mech_vent_6h": "Mechanical ventilation within 6h",
    "ph_worst": "Worst pH",
    "ph_n_meas": "pH measurement count",
    "spo2_h6": "SpO2 at hour 6",
    "bmi": "BMI",
    "paco2_worst": "Worst PaCO2",
    "potassium_n_meas": "Potassium measurement count",
    "chloride_worst": "Worst chloride"
}

featureInputTypeMap = {
    "lactate_n_meas": "number",
    "pao2_worst": "number",
    "mech_vent_6h": "select",
    "ph_worst": "number",
    "ph_n_meas": "number",
    "spo2_h6": "number",
    "bmi": "number",
    "paco2_worst": "number",
    "potassium_n_meas": "number",
    "chloride_worst": "number"
}

featureOptionMap = {
    "mech_vent_6h": [0, 1]
}
