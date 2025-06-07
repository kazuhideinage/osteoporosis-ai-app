import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import io

# --- モデルとラベルエンコーダを読み込む ---
model = joblib.load("rf_model.joblib")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
class_names = le.classes_

st.title("骨粗鬆症薬選択支援AIアシスタント")
st.markdown("---")

st.header("📝 患者情報の入力")

# --- 入力フォーム ---
age = st.number_input("年齢", min_value=40, max_value=100, value=75)
sex = st.radio("性別", ("男", "女"))
lumbar_yam = st.number_input("腰椎YAM値（%）", min_value=10.0, max_value=120.0, value=60.0)
femoral_yam = st.number_input("大腿骨頸部YAM値（%）", min_value=10.0, max_value=120.0, value=58.0)
tracp5b = st.number_input("TRACP-5b値 (mU/dL)", min_value=0.0, value=400.0)
egfr = st.number_input("eGFR値 (mL/min)", min_value=5.0, value=65.0)
bone_fracture = st.radio("骨折歴", ("あり", "なし", "不明"))
steroid = st.radio("ステロイド使用", ("あり", "なし", "不明"))
diabetes = st.radio("糖尿病既往", ("あり(2型)", "なし", "不明"))

if st.button("💊 薬剤を推奨する"):
    # --- 入力データをモデルに合わせて整形 ---
    input_data = pd.DataFrame({
        "年齢": [age],
        "腰椎YAM": [lumbar_yam],
        "大腿骨頸部YAM": [femoral_yam],
        "TRACP5b": [tracp5b],
        "eGFR": [egfr],
        "性別_男": [1 if sex == "男" else 0],
        "骨折歴_なし": [1 if bone_fracture == "なし" else 0],
        "骨折歴_不明": [1 if bone_fracture == "不明" else 0],
        "ステロイド使用_なし": [1 if steroid == "なし" else 0],
        "ステロイド使用_不明": [1 if steroid == "不明" else 0],
        "糖尿病既往_あり(2型)": [1 if diabetes == "あり(2型)" else 0],
        "糖尿病既往_なし": [1 if diabetes == "なし" else 0],
        "糖尿病既往_不明": [1 if diabetes == "不明" else 0]
    })

    # --- 予測 ---
    proba = model.predict_proba(input_data)[0]
    top_index = np.argmax(proba)
    top_drug = class_names[top_index]

    st.success(f"\n✅ 推奨薬剤：**{top_drug}**（確率：{proba[top_index]*100:.1f}%）")

    # --- 全カテゴリの確率を表示 ---
    st.subheader("📊 推奨確率（全薬剤）")
    prob_df = pd.DataFrame({"薬剤": class_names, "確率（%）": proba * 100})
    st.table(prob_df.sort_values(by="確率（%）", ascending=False).reset_index(drop=True))

    # --- SHAPで特徴量の寄与を可視化 ---
    st.subheader("🧠 推奨根拠（SHAPによる説明）")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_values[top_index][0], 
                                          base_values=explainer.expected_value[top_index],
                                          data=input_data.iloc[0],
                                          feature_names=input_data.columns.tolist()),
                        max_display=10, show=False)
    st.pyplot(fig)

    st.markdown("---")
    st.caption("※ 本結果はAIモデルによる予測であり、臨床判断を置き換えるものではありません")

    # --- PDFレポート出力機能 ---
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

    # PDFバイト出力
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    b64 = base64.b64encode(pdf_output.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📥 PDFレポートをダウンロード</a>'
    st.markdown(href, unsafe_allow_html=True)
