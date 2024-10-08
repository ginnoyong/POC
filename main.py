# Set up and run this Streamlit App
import streamlit as st
from helper_functions import llm # <--- This is the helper function that we have created 🆕


# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Streamlit App")

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    st.toast(f"User Input Submitted - {user_prompt}")
    response = llm.get_completion(user_prompt) # <--- This calls the helper function that we have created 🆕
    st.write(response) # <--- This displays the response generated by the LLM onto the frontend 🆕
    print(f"User Input is {user_prompt}")