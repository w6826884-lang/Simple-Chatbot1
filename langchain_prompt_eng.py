import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# API key: .env locally, Streamlit secrets when deployed
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_key = None

if not groq_key:
    st.error("No Groq API key found. Set GROQ_API_KEY in your .env or Streamlit secrets.")
    st.stop()

model = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_key, temperature=0.3)

template = PromptTemplate(
    template="""Please summarise the research paper named: {paper_input}.
Explanation style: {select_explanation_type}
Length of response: {input_length}

Style rules:
- If the style is Mathematical: include the detailed formulae and their derivations.
- If the style is Beginner Friendly: use very simple language that anyone can follow.
- If the style is Technical: focus on architecture, methods and results in precise terms.
- If the style is Code-Heavy: include pseudocode or Python snippets illustrating the key ideas.

If you are not sure the paper exists, say so instead of inventing details.""",
    input_variables=["paper_input", "select_explanation_type", "input_length"],
)

st.header("Research Tool")

paper_input = st.text_input("Research paper name")
select_explanation_type = st.selectbox(
    "Explanation style",
    ["Beginner Friendly", "Technical", "Code-Heavy", "Mathematical"],
)
input_length = st.selectbox(
    "Length of explanation",
    ["1-2 paragraphs", "3-4 paragraphs", "Long and comprehensive"],
)

if st.button("Reply"):
    if not paper_input.strip():
        st.warning("Please enter a research paper name.")
    else:
        chain = template | model
        with st.spinner("Thinking..."):
            result = chain.invoke({
                "paper_input": paper_input,
                "select_explanation_type": select_explanation_type,
                "input_length": input_length,
            })
        st.write(result.content)
