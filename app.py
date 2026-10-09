import streamlit as st
from pipeline import master_router

st.set_page_config(page_title="PharmaSense AI", page_icon="🧪")
st.title("🧪 PharmaSense AI")
st.caption("Multi-agent GenAI assistant for pharma R&D and clinical ops")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about trials, compounds, safety, or literature..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = master_router(question)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})