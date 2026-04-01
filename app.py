import warnings

warnings.filterwarnings("ignore")

import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

from config import (
    appTitle,
    resultTitle,
    systemDesc,
    moduleDesc,
    footerText,
    softwareVersion,
    bundlePath,
    templatePath,
    batchResultPath,
)
from predict_utils import (
    loadModelBundle,
    prepareInputData,
    predictSingle,
    getRisk,
    buildResultTable,
    manualInputToDf,
    predictBatchCases,
    saveBatchResult,
)


def showApp():
    if st is None:
        raise ImportError("当前环境没有安装 streamlit")

    st.set_page_config(
        page_title=appTitle,
        page_icon="🩺",
        layout="wide"
    )

    if "resetId" not in st.session_state:
        st.session_state["resetId"] = 0
    if "batchDf" not in st.session_state:
        st.session_state["batchDf"] = None
    if "lastProb" not in st.session_state:
        st.session_state["lastProb"] = None
    if "lastRisk" not in st.session_state:
        st.session_state["lastRisk"] = None
    if "lastDf" not in st.session_state:
        st.session_state["lastDf"] = None

    try:
        bundle = loadModelBundle(bundlePath)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

    model = bundle["model"]
    featureNames = bundle["featureNames"]
    fillValues = bundle["fillValues"]
    threshold = bundle["threshold"]
    pageTitle = bundle.get("appTitle", appTitle)
    pageResultTitle = bundle.get("resultTitle", resultTitle)
    nameMap = bundle.get("featureNameMap", {})
    typeMap = bundle.get("featureInputTypeMap", {})
    optionMap = bundle.get("featureOptionMap", {})

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 1.0rem;
            max-width: 100%;
        }

        .main-title {
            font-size: 28px;
            font-weight: 800;
            color: #133a6f;
            margin-top: 0;
            margin-bottom: 8px;
            line-height: 1.4;
            word-break: break-word;
            white-space: normal;
            overflow: visible;
        }

        .sub-text {
            font-size: 14px;
            color: #5c6b7a;
            line-height: 1.6;
            margin-bottom: 8px;
            word-break: break-word;
        }

        .card {
            background: #ffffff;
            padding: 16px;
            border-radius: 16px;
            border: 1px solid #e7eef6;
            box-shadow: 0 4px 10px rgba(16, 35, 58, 0.04);
            margin-bottom: 12px;
        }

        .result-box {
            background: #ffffff;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #dbe7f3;
            box-shadow: 0 4px 10px rgba(16, 35, 58, 0.04);
        }

        .result-label {
            font-size: 14px;
            color: #66788a;
            margin-bottom: 4px;
            font-weight: 600;
        }

        .result-value {
            font-size: 38px;
            font-weight: 800;
            color: #133a6f;
            margin-bottom: 10px;
        }

        .footer-box {
            font-size: 13px;
            color: #5c6b7a;
            margin-top: 8px;
            line-height: 1.8;
        }

        .small-note {
            font-size: 13px;
            color: #617284;
            line-height: 1.8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">创伤后脓毒症早期风险预测系统 V1.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">用于创伤后脓毒症早期风险预测，支持单例预测和批量预测。</div>',
                unsafe_allow_html=True)

    predictBtn = False
    resetBtn = False

    leftCol, rightCol = st.columns(2)

    with leftCol:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("手动输入")

        inputValues = {}
        col1, col2 = st.columns(2)

        for idx, feature in enumerate(featureNames):
            label = nameMap.get(feature, feature)
            inputType = typeMap.get(feature, "number")
            defaultValue = fillValues.get(feature, 0)
            widgetKey = f"{feature}_{st.session_state['resetId']}"

            targetBox = col1 if idx % 2 == 0 else col2

            with targetBox:
                if inputType == "select":
                    options = optionMap.get(feature, [0, 1])
                    if defaultValue not in options:
                        defaultValue = options[0]
                    defaultIndex = options.index(defaultValue)

                    inputValues[feature] = st.selectbox(
                        label,
                        options,
                        index=defaultIndex,
                        key=widgetKey
                    )
                else:
                    inputValues[feature] = st.number_input(
                        label,
                        value=float(defaultValue),
                        key=widgetKey
                    )

        btn1, btn2 = st.columns(2)
        with btn1:
            predictBtn = st.button("开始预测", use_container_width=True)
        with btn2:
            resetBtn = st.button("重置输入", use_container_width=True)

        if resetBtn:
            st.session_state["resetId"] += 1
            st.session_state["batchDf"] = None
            st.session_state["lastProb"] = None
            st.session_state["lastRisk"] = None
            st.session_state["lastDf"] = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with rightCol:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(pageResultTitle)

        if predictBtn:
            try:
                df = manualInputToDf(inputValues, featureNames)
                x = prepareInputData(df, featureNames, fillValues)

                prob = predictSingle(model, x)
                risk = getRisk(prob, threshold)

                st.session_state["lastProb"] = prob
                st.session_state["lastRisk"] = risk
                st.session_state["lastDf"] = x.copy()
            except Exception as e:
                st.error(f"单例预测失败: {e}")

        currentProb = st.session_state["lastProb"]
        currentRisk = st.session_state["lastRisk"]

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        if currentProb is None:
            st.markdown('<div class="result-label">预测概率</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-value">--</div>', unsafe_allow_html=True)
            st.info("请先输入变量。")
        else:
            st.markdown('<div class="result-label">预测概率</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">{currentProb * 100:.2f}%</div>', unsafe_allow_html=True)

            if currentRisk == "高风险":
                st.warning(f"风险分层结果: {currentRisk}")
            else:
                st.success(f"风险分层结果: {currentRisk}")

            if currentProb >= 0.80:
                st.write("高风险提示。")
            elif currentProb >= 0.50:
                st.write("中等风险提示。")
            elif currentProb >= 0.20:
                st.write("一般风险提示。")
            else:
                st.write("低风险提示。")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="small-note">结果仅供研究参考。</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["lastDf"] is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("本次输入变量")
        df = buildResultTable(st.session_state["lastDf"], featureNames)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("下载模板")
    st.markdown(
        '<div class="small-note">下载模板后按列名填写。</div>',
        unsafe_allow_html=True
    )

    try:
        with open(templatePath, "rb") as f:
            st.download_button(
                label="下载 input_template.csv",
                data=f,
                file_name="input_template.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.warning("未找到模板文件，请先生成模板。")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("批量预测")
    st.markdown(
        '<div class="small-note">上传CSV文件后进行批量预测。</div>',
        unsafe_allow_html=True
    )

    uploadedFile = st.file_uploader("上传CSV文件", type=["csv"])

    if uploadedFile is not None:
        try:
            df = pd.read_csv(uploadedFile)
            out = predictBatchCases(
                model=model,
                df=df,
                featureNames=featureNames,
                fillValues=fillValues,
                threshold=threshold
            )

            st.session_state["batchDf"] = out
            saveBatchResult(out, batchResultPath)

            st.success("批量预测完成")
            st.dataframe(out, use_container_width=True, hide_index=True)
        except Exception as e:
            st.session_state["batchDf"] = None
            st.error(f"批量预测失败: {e}")

    if st.session_state["batchDf"] is not None:
        csvData = st.session_state["batchDf"].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载批量预测结果",
            data=csvData,
            file_name="batch_prediction_results.csv",
            mime="text/csv"
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="footer-box">{footerText}<br>当前版本: {softwareVersion}</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    showApp()
