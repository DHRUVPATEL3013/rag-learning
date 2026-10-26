from langchain_core.output_parsers import StrOutputParser
from prompts.prompts import rewrite_prompt
from llm import rewrite_prompt,llm


rewrite_chain= rewrite_prompt | llm | StrOutputParser()


original="what are the types of machine learning?"

rewritten=rewrite_chain.invoke({
  "question":original
})

print(original)
print("-------")
print(rewritten)