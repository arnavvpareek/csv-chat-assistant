"""Streamlit UI: upload a CSV, ask questions in a chat, see the answer
alongside the pandas code that produced it."""

import pandas as pd
import streamlit as st

from agent import ask_agent

st.set_page_config(
    page_title="CSV Chat Assistant",
    page_icon="📊",
    layout="centered",
)

# ---------- Sidebar: upload + dataset facts ----------
with st.sidebar:
    st.header("📊 CSV Chat Assistant")
    st.caption("Ask questions about any CSV in plain English. Answers come "
               "with the pandas code that produced them.")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("That file couldn't be parsed as a CSV. "
                     "Please check the file and try again.")
        else:
            if df.empty:
                st.error("This CSV has no rows. Upload a file with data.")
                df = None

    if df is not None:
        st.success(f"**{uploaded.name}** loaded")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        with st.expander("Column details"):
            st.dataframe(
                pd.DataFrame({
                    "column": df.columns,
                    "type": [str(t) for t in df.dtypes],
                    "missing": df.isna().sum().values,
                }),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    st.caption("Built with Streamlit · LangChain · Groq (Llama 3.3 70B). "
               "The agent loop is hand-built — no `langchain_experimental`.")

# ---------- Reset chat when the dataset changes ----------
current_file = uploaded.name if uploaded is not None else None
if st.session_state.get("file_name") != current_file:
    st.session_state.file_name = current_file
    st.session_state.messages = []

# ---------- Main area ----------
if df is None:
    st.title("Chat with your data 📊")
    st.markdown(
        "Upload a CSV in the sidebar, then ask questions like:\n"
        "- *What's the average rating by genre?*\n"
        "- *Which 5 titles are the most recent?*\n"
        "- *How many entries per country?*\n\n"
        "Every answer shows the **pandas code** that computed it — "
        "no black box, no hallucinated numbers."
    )
    st.info("⬅ Start by uploading a CSV file in the sidebar.")
    st.stop()

st.title(f"Chatting with `{uploaded.name}`")
with st.expander("Preview data", expanded=not st.session_state.messages):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Showing first 10 of {len(df):,} rows.")

# ---------- Replay chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("code"):
            st.code(msg["code"], language="python")
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True)

# ---------- Starter questions (shown only before the first message) ----------
question = None
if not st.session_state.messages:
    st.caption("Try one of these to get started:")
    starters = [
        "Give me an overview of this dataset",
        "Which columns have missing values?",
        "Show the top 5 rows sorted by the most interesting numeric column",
    ]
    cols = st.columns(len(starters))
    for col, s in zip(cols, starters):
        if col.button(s, use_container_width=True):
            question = s

typed = st.chat_input("Ask a question about your data…")
if typed:
    question = typed

# ---------- Handle a new question ----------
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Writing and running pandas code…"):
            try:
                out = ask_agent(df, question)
            except Exception as e:
                error_text = (
                    "I couldn't compute an answer for that. Here's what "
                    f"went wrong:\n\n```\n{e}\n```\n\n"
                    "Try rephrasing the question or referring to columns "
                    "by their exact names."
                )
                st.markdown(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text}
                )
            else:
                st.markdown(out["answer"])
                st.code(out["code"], language="python")
                table = None
                result = out["result"]
                if isinstance(result, pd.Series):
                    table = result.to_frame()
                elif isinstance(result, pd.DataFrame):
                    table = result
                if table is not None:
                    st.dataframe(table, use_container_width=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": out["answer"],
                    "code": out["code"],
                    "table": table,
                })
