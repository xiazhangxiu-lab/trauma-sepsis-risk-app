from pathlib import Path

import joblib
import pandas as pd


def makeBatchTemplate(featureNames, savePath, fillValues=None, nRows=5):
    savePath = Path(savePath)

    if not isinstance(featureNames, list) or len(featureNames) == 0:
        raise ValueError("featureNames为空")

    rows = []
    for _ in range(nRows):
        row = {}
        for name in featureNames:
            if fillValues is not None and name in fillValues:
                row[name] = fillValues[name]
            else:
                row[name] = ""
        rows.append(row)

    df = pd.DataFrame(rows, columns=featureNames)
    df.to_csv(savePath, index=False, encoding="utf-8-sig")
    return df


def loadModelBundle(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件: {path}")

    bundle = joblib.load(path)

    needKeys = ["model", "featureNames", "fillValues", "threshold"]
    missKeys = [key for key in needKeys if key not in bundle]
    if missKeys:
        raise KeyError(f"模型文件缺少字段: {missKeys}")

    return bundle


def toDataFrame(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, dict):
        return pd.DataFrame([data])

    raise TypeError("输入只支持DataFrame或dict")


def prepareInputData(df, featureNames, fillValues):
    df = toDataFrame(df)

    missCols = [name for name in featureNames if name not in df.columns]
    if missCols:
        raise ValueError(f"缺少字段: {missCols}")

    x = df[featureNames].copy()

    for name in featureNames:
        x[name] = pd.to_numeric(x[name], errors="coerce")
        x[name] = x[name].fillna(fillValues.get(name, 0))

    return x


def getRisk(prob, threshold):
    return "高风险" if prob >= threshold else "低风险"


def predictSingle(model, df):
    if len(df) != 1:
        raise ValueError("单例预测只能输入 1 行数据")

    prob = model.predict_proba(df)[:, 1][0]
    return float(prob)


def predictBatchCases(model, df, featureNames, fillValues, threshold):
    x = prepareInputData(df, featureNames, fillValues)
    prob = model.predict_proba(x)[:, 1]

    out = x.copy()
    out["predictedProbability"] = prob
    out["riskGroup"] = out["predictedProbability"].apply(lambda value: getRisk(value, threshold))
    return out


def buildResultTable(df, featureNames):
    df = toDataFrame(df)

    if len(df) != 1:
        raise ValueError("结果展示只能输入 1 行数据")

    out = pd.DataFrame({
        "variable": featureNames,
        "value": df.iloc[0][featureNames].values
    })
    return out


def saveBatchResult(df, path):
    path = Path(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def manualInputToDf(inputValues, featureNames):
    df = pd.DataFrame([inputValues])
    return df[featureNames]
