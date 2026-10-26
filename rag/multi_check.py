from detoxify import Detoxify
from langchain_core.output_parsers import StrOutputParser
from rag.llm import llm
from rag.prompts.prompts import grounded_prompt,relevant_prompt,prompt


model=Detoxify("original")
ground_check=grounded_prompt | llm | StrOutputParser()
relevance_check=relevant_prompt | llm | StrOutputParser()

rag_chain=prompt | llm | StrOutputParser()

def is_safe(answer:str,threshold:float=0.5):
  scores=model.predict(answer)

  for category ,score in scores.items():
    if score > threshold:
      return False
  return True

def is_grounded(context,question,answer):
    grounded = ground_check.invoke(
    {
        "context":context,
        "question":question,
        "answer":answer
    }
         ).strip().upper().startswith("YES")

    return grounded
       
       


def is_relevance(question,answer):
    relevant=relevance_check.invoke(
                {
                    "question":question,
                    "answer":answer
                }
            ).strip().upper().startswith("YES")
    return relevant
    




   

   





