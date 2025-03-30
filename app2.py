from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe

# Connect to local Llama 3 via Ollama
model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3"
)

st.title("Data analysis with Pandas AI and Llama 3")

input_files = st.file_uploader("Upload your CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True)

if input_files:
    selected_file = st.selectbox("Select a file", [file.name for file in input_files])
    selected_index = [file.name for file in input_files].index(selected_file)

    st.info("File uploaded successfully")
    file_extension = selected_file.split('.')[-1]

    st.text("Head (3)")

    try:
        if file_extension == 'csv':
            data = pd.read_csv(input_files[selected_index])
        elif file_extension == 'xlsx':
            data = pd.read_excel(input_files[selected_index])

        st.dataframe(data.head(3), use_container_width=True)

        df = SmartDataframe(
            data,
            config={"llm": model, "enable_cache": False}
        )

        prompt = st.text_area("What do you want to ask?")

        if st.button("Ask"):
            if prompt:
                with st.spinner("Generating Response..."):
                    st.write(df.chat(prompt))

    except Exception as e:
        st.error(f"Error: {e}")

