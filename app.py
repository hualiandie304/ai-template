import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ======================
# 路径设置
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "AI模型")
model_path = os.path.join(model_dir, "rlc_model.pkl")
upload_dir = os.path.join(BASE_DIR, "upload_history")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(upload_dir, exist_ok=True)

# ======================
# 初始化模型
# ======================
if not os.path.exists(model_path):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    joblib.dump(model, model_path)

# ======================
# 页面标题
# ======================
st.title("⚡ RLC 异常检测 AI 平台")
st.subheader("上传Excel学习 → 输入数据判断异常")

tab1, tab2 = st.tabs(["📚 AI学习（上传Excel）", "🔍 判断数据是否异常"])

# ======================
# 选项卡1：训练AI
# ======================
with tab1:
    st.subheader("上传训练数据")
    
    # 新增：让用户输入自己的名字
    uploader_name = st.text_input("请输入你的名字（用于文件命名）：")
    uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx"])

    if uploaded_file is not None and uploader_name:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        if st.button("开始训练AI"):
            try:
                # ---------------
                # 1. 按规则生成文件名：名字_时间戳.xlsx
                # ---------------
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{uploader_name}_{time_str}.xlsx"
                save_path = os.path.join(upload_dir, filename)

                # ---------------
                # 2. 只保存一次文件
                # ---------------
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # ---------------
                # 3. 训练模型
                # ---------------
                X = df[["频率(Hz)", "电流(A)"]].values
                y = df["是否异常"].values

                model = joblib.load(model_path)
                model.fit(X, y)
                joblib.dump(model, model_path)

                st.success(f"✅ 训练完成！文件已保存为：{filename}")
            except Exception as e:
                st.error(f"训练失败：{e}")
    elif uploaded_file is not None and not uploader_name:
        st.warning("⚠️ 请先输入你的名字！")

# ======================
# 选项卡2：预测
# ======================
with tab2:
    st.subheader("手动输入数据判断")
    freq = st.number_input("频率 (Hz)", value=3000.0)
    current = st.number_input("电流 (A)", value=0.05)

    if st.button("判断是否异常"):
        try:
            model = joblib.load(model_path)
            res = model.predict([[freq, current]])

            if res[0] == 1:
                st.error("⚠️ AI 判断：该数据为异常数据")
            else:
                st.success("✅ AI 判断：该数据为正常数据")
        except:
            st.warning("⚠️ 请先训练模型")
