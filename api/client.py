import requests
import streamlit as st

# ---- Essay (Llama3) ----
def get_essay_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay",
        json={'topic': input_text}
    )
    return response.json()['essay']

# ---- Poem (Llama2) ----
def get_poem_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem",
        json={'topic': input_text}
    )
    return response.json()['poem']

# ---- Streamlit UI ----
st.title('LangChain Demo With Llama Models 🦙')

input_text = st.text_input("Write an essay through (Llama3)")
input_text1 = st.text_input("Write a poem through (Llama2)")

if input_text:
    st.subheader("📝 Essay Result:")
    st.write(get_essay_response(input_text))

if input_text1:
    st.subheader("🎵 Poem Result:")
    st.write(get_poem_response(input_text1))
