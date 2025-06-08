
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("骨粗鬆症薬選択支援AIアシスタント（SHAPなし軽量版）")
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

    features = ["年齢", "投与前 腰椎 YAM値(%)", "投与前 Tracp 5b値(mU/dL)", 
                "投与前 eGFR値(mL/min)", "投与前 Ca値(mg/dl)", "投与前 ALB値(g/dl)"]
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

        st.success(f"✅ 推奨薬剤：{top_drug}（確率：{proba[top_index]*100:.1f}%）")

        st.subheader("📊 推奨確率（全薬剤）")
        prob_df = pd.DataFrame({"薬剤": class_names, "確率（%）": proba * 100})
        st.table(prob_df.sort_values(by="確率（%）", ascending=False).reset_index(drop=True))
