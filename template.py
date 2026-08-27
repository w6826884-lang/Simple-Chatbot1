from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template = """Please summarise the paper names {paper_input}.
      Explanation type: {select_explanation_type} the length of response :{input_length}.
      If explanation type =  mathematical include detailed formulaes and the derivations used in this derivation.
      If explanation type - Beginner Friendly summarise the paper in a very easy to understand language which can be followed by everyone.""",
      input_variables=['paper_input', 'select_explanation_type', 'input_length']
)
template.save('template.json')