from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core. prompts import PromptTemplate
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro-0813",
    task='text-generation'
)
# template
template = PromptTemplate(
    template = """Please summarise the paper names {paper_input}.
      Explanation type: {select_explanation_type} the length of response :{input_length}.
      If explanation type =  mathematical include detailed formulaes and the derivations used in this derivation.
      If explanation type - Beginner Friendly summarise the paper in a very easy to understand language which can be followed by everyone.""",
      input_variables=['paper_input', 'select_explanation_type', 'input_length']
)
model = ChatHuggingFace(llm=llm)
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