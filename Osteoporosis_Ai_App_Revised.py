
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io

st.title("骨粗鬆症薬選択支援AIアシスタント（CSVからモデル再学習）")
st.markdown("---")

st.header("📤 CSVデータのアップロード（4. Bonet 1年後ろ向き.csv）")
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="cp932")

    st.success("✅ データ読み込み完了")

    st.header("📝 患者情報の入力")
    age = st.number_input("年齢", min_value=40, max_value=100, value=75)
    sex = st.radio("性別", ("男", "女"))
    yam = st.number_input("腰椎YAM値（%）", min_value=10.0, max_value=120.0, value=60.0)
    tracp5b = st.number_input("TRACP-5b値 (mU/dL)", min_value=0.0, value=400.0)
    egfr = st.number_input("eGFR値 (mL/min)", min_value=5.0, value=65.0)
    ca = st.number_input("Ca値 (mg/dL)", min_value=6.0, max_value=12.0, value=9.5)
    alb = st.number_input("Alb値 (g/dL)", min_value=2.0, max_value=5.0, value=4.0)

    features = ["年齢", "投与前 腰椎 YAM値(%)", "投与前 Tracp 5b値(mU/dL)", "投与前 eGFR値(mL/min)", "投与前 Ca値(mg/dl)", "投与前 ALB値(g/dl)"]
    target = "使用骨粗鬆症薬名①(メイン)"

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y_encoded)
    class_names = le.classes_

    st.success("✅ モデル学習が完了しました")

    if st.button("💊 薬剤を推奨する"):
        input_data = pd.DataFrame({
            "年齢": [age],
            "投与前 腰椎 YAM値(%)": [yam],
            "投与前 Tracp 5b値(mU/dL)": [tracp5b],
            "投与前 eGFR値(mL/min)": [egfr],
            "投与前 Ca値(mg/dl)": [ca],
            "投与前 ALB値(g/dl)": [alb]
        })

        proba = model.predict_proba(input_data)[0]
        top_index = np.argmax(proba)
        top_drug = class_names[top_index]

        st.success(f"✅ 推奨薬剤：**{top_drug}**（確率：{proba[top_index]*100:.1f}%）")

        st.subheader("📊 推奨確率（全薬剤）")
        prob_df = pd.DataFrame({"薬剤": class_names, "確率（%）": proba * 100})
        st.table(prob_df.sort_values(by="確率（%）", ascending=False).reset_index(drop=True))

        st.subheader("🧠 推奨根拠（SHAPによる説明）")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # SHAPの出力形式に対応
        if isinstance(shap_values, list):  # 多クラス分類
            sv = shap_values[top_index][0]
            base_val = explainer.expected_value[top_index]
        else:  # 単クラス or 2クラス分類
            sv = shap_values[0]
            base_val = explainer.expected_value

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap.Explanation(
            values=sv,
            base_values=base_val,
            data=input_data.iloc[0],
            feature_names=input_data.columns.tolist()
        ), max_display=10, show=False)
        st.pyplot(fig)

        st.subheader("📄 レポート出力 (PDF)")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="骨粗鬆症薬選択支援AI レポート", ln=1, align="C")
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"推奨薬剤：{top_drug}（確率：{proba[top_index]*100:.1f}%）", ln=1)
        pdf.ln(5)
        pdf.cell(200, 10, txt="入力情報：", ln=1)
        for col in input_data.columns:
            val = input_data[col].values[0]
            pdf.cell(200, 8, txt=f"{col}: {val}", ln=1)

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        b64 = base64.b64encode(pdf_output.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📥 PDFレポートをダウンロード</a>'
        st.markdown(href, unsafe_allow_html=True)
