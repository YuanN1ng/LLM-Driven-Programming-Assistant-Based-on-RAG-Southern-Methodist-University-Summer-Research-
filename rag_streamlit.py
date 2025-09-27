# -*- coding: utf-8 -*-
"""

conda activate code_rag
streamlit run rag_streamlit.py
"""

#copy a image 非第一但由G纠正
#randomly generate a video 不存在
#scatter plot 注释不够清晰
#散点图 gcb无法理解中文
#plot after clearing current plot 注释干扰但由G点明（使用会清空当前？需要先清空再使用？）
#delete the same word that alaready show up in the front 排第四
import streamlit as st
import d_search, call_olla

st.set_page_config(page_title="Local RAG Demo", page_icon="💡", layout="wide")

# ---- 侧边栏 ----
st.sidebar.header("🔧 Setting")
model = st.sidebar.radio(
    "Select the embedding model",
    ["Gemini", "GraphCodeBert"],
    index=0,
    horizontal=True,
)

# ---- 主区 ----
st.title("🌾 A Code Search Framework based on Retrival Augmented Generation (RAG)")

query = st.text_area("📝 Input natural language description", placeholder="e.g. copy a image")
run_btn = st.button("⚡ Retrival Augmented Generation", type="primary")

if run_btn and query.strip():
    # ---------- Retrieval ----------
    with st.spinner("🔍 Searching for relevant code..."):
        docs = (d_search.gemini_query if model == "gemini"
                else d_search.graphcodebert_query)(query, topk=5)

    st.subheader("🔎 Search results (Top-5)")
    for i, doc in enumerate(docs, 1):
        with st.expander(f"{i}. Code preview", expanded=(i == 1)):
            st.code(doc, language="python")

    # ---------- Generation ----------
    with st.spinner("🤖 Using Qwen3-14B to generate the answer…"):
        context = "\n---\n".join(docs)
        prompt = (
            "From now on, you just need to focus on the latest input from the user and ignore all the previous conversation content. "
            "You are an experienced coding expert. "
            "Please, based on the user's natural language description and the five relevant code snippets retrieved, first identify the code snippet that best meets the user's needs among these five."
            "If none of the five retrieved codes are suitable, it should be explained the situation and the reasons, and at the same time, appropriate codes outside the retrieved ones should also be provided."
            "You also need to teach user how to use it."
            "If the context is insufficient to provide an answer, it should be clearly stated without any exaggeration.\n\n"
            "Natural language description from user: \n"
            f"{query}\n\n"
            "The five code snippets retrieved:\n"
            f"{context}\n\n"
            "Answer:"
        )
        answer = call_olla.call_ollama(prompt, model="qwen3:14b")

    st.subheader("💡 Response from Qwen3-14B")
    st.markdown(answer)
elif run_btn:
    st.warning("Please enter your query and then click the button~")