import time

import streamlit as st
from pipeline_clean import master_router

st.set_page_config(page_title="PharmaSense AI", page_icon="🧪")
st.title("🧪 PharmaSense AI")
st.caption("Multi-agent GenAI assistant for pharma R&D and clinical ops")

# Visible safety notice
st.info(
    "PharmaSense AI supports research workflows. Its outputs are not medical "
    "advice and must be reviewed by a qualified human."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("latency_seconds") is not None:
            st.caption(f"Response time: {msg['latency_seconds']:.2f} seconds")
        if msg.get("fallback"):
            st.warning("Fallback response: the system could not provide a verified answer.")

if question := st.chat_input("Ask about trials, compounds, safety, or literature..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    start_time = time.perf_counter()
    fallback = False

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = master_router(question)

                # Guard against empty or non-text responses.
                if not isinstance(answer, str) or not answer.strip():
                    answer = (
                        "I couldn't generate a usable answer. Please rephrase your "
                        "question or try again. No verified answer was generated."
                    )
                    fallback = True

            except Exception:
                # Avoid exposing stack traces, API details, or secrets to users.
                answer = (
                    "The AI service encountered an error and could not complete "
                    "your request. Please try again later. No verified answer was generated."
                )
                fallback = True

    latency_seconds = time.perf_counter() - start_time

    st.write(answer)
    st.caption(f"Response time: {latency_seconds:.2f} seconds")
    if fallback:
        st.warning("Fallback response: the system could not provide a verified answer.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "latency_seconds": latency_seconds,
            "fallback": fallback,
        }
    )
