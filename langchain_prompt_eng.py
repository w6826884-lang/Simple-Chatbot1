import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_key:
    st.error("No Groq API key found. Set GROQ_API_KEY in your .env or Streamlit secrets.")
    st.stop()




# template
template = PromptTemplate(
    template = """Please summarise the paper names {paper_input}.
      Explanation type: {select_explanation_type} the length of response :{input_length}.
      If explanation type =  mathematical include detailed formulaes and the derivations used in this derivation.
      If explanation type - Beginner Friendly summarise the paper in a very easy to understand language which can be followed by everyone.""",
      input_variables=['paper_input', 'select_explanation_type', 'input_length']
)
model = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_key, temperature=0.3)
paper_input = st.text_input("Select Research Paper Name")
select_explanation_type = st.selectbox("Select the explanation style",["Beginner_Friendly","Technical","Code-Heavy","Mathematical"])
input_length = st.selectbox("Select the length of explanation",["1-2 paragraph","3-4 paragraph","long and comprehensive"])
st.header('Research Tool')
user_input = template
if st.button("Reply"):
    chain = template | model
    result = chain.invoke({
        "paper_input":paper_input,
        "select_explanation_type":select_explanation_type,
        "input_length":input_length 
    })
    st.write(result.content)
else:
    st.warning("Please enter a prompt.")
