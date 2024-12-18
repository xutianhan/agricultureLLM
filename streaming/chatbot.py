import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict/"

# 显示页面标题
st.title("农业问答系统")

# 输入框，获取用户输入
user_input = st.text_input("请输入你的问题：", "")

# 输出框，展示模型的回复
response_placeholder = st.empty()

if user_input:
    # 显示加载动画
    response_placeholder.markdown("正在生成回答，请稍等...")

    # 向FastAPI后端发送请求获取回答
    response = requests.post(API_URL, json={"input_text": user_input}, stream=True)

    # 逐字显示模型的回复
    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                response_placeholder.markdown(f"<pre>{line}</pre>", unsafe_allow_html=True)
    else:
        response_placeholder.markdown("发生了错误，请稍后再试。")