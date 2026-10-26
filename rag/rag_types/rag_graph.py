from typing  import TypedDict,List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph,END
import pickle
from rag.config.config import parent_store
from rag.context.llm_context import fit_to_budget,reorder_docs
from rag.llm import llm,format_docs
from rag.prompts.prompts import prompt,grade_prompt,graph_rewrite_prompt,decide_prompt
from rag.retriever.dense_retriever import vectorstore
from rag.retriever.retriever import final_retriver



class RAGState(TypedDict):
  question: str
  documents : List[Document]
  answer: str
  grade:str
  retries:int
  needs_retrieval:str

grade_chain=grade_prompt | llm | StrOutputParser()
rewrite_chain=graph_rewrite_prompt | llm | StrOutputParser()
decide_chain=decide_prompt | llm | StrOutputParser()

def decide_retrieval_node(state:RAGState):
    needs=decide_chain.invoke({
      "question":state["question"]
   }).strip().upper()
    print(f"[DECIDE] {needs}")
    return {"needs_retrieval":needs}
def retriever_node(state: RAGState):
   #  docs = final_retriver.invoke(state['question'])
    results=vectorstore.similarity_search_with_relevance_scores(state['question'],k=5)
    docs=[d for d,s in results]
    scores=[s for d,s in results]
    best_scores=max(scores)

    if best_scores < 0.5 :
       return {"documents":[],"grade":"IRRELEVANT"}


    return {"documents": docs}

def grade_node(state:RAGState):
   if state.get("grade") == "IRRELEVANT":
      return {"grade":"IRRELEVANT"}

   context = format_docs(state["documents"])
   grade=grade_chain.invoke({
      "question":state['question'],
      "documents":context
   }).strip().upper()
   print(f"grade : {grade}")

   return {"grade":"YES" if "YES" in grade else "NO"}

def rewrite_node(state:RAGState):
   new_q=rewrite_chain.invoke({
      "question":state['question']
   }).strip()
   print(f"rewritten question : {new_q}")
   return {"question":new_q,"retries":state.get("retries",0) + 1}
def generate_node(state:RAGState):
  if state.get("documents"):
    context=format_docs(state['documents'])
    answer=(prompt | llm | StrOutputParser()).invoke({
      "context": context,
          "question": state["question"],
    })
  else:
     answer=llm.invoke(state["question"]).content
  return {"answer":answer}


def decide_next(state:RAGState):
   if state["grade"] == "YES":
      return "generate"
   if state["grade"] == "IRRELEVANT":
      return "no_answer"
   if state.get("retries",0) >=2:
      return "generate"          # genearte after 2 retrives
   return "rewrite"

def no_answer_node(state:RAGState):
   return {"answer":"I don't know based on the provided context."}
graph = StateGraph(RAGState)

graph.add_node("retrieve",retriever_node)
graph.add_node("grade",grade_node)
graph.add_node("rewrite", rewrite_node)  # Add node for rewrite
graph.add_node("generate",generate_node)
graph.add_node("decide_retrieval", decide_retrieval_node)  # Add node for decide_retrieval
graph.add_node("no_answer", no_answer_node)  # Add node for no_answer

graph.set_entry_point("decide_retrieval")
graph.add_conditional_edges("decide_retrieval",
                  lambda state : state["needs_retrieval"],
                  {
                     "YES":"retrieve",
                     "NO":"generate"
                  }
                            )
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade",decide_next,{
    "generate": "generate",
    "rewrite": "rewrite",
    "no_answer":"no_answer"

})
graph.add_edge("rewrite","retrieve")
graph.add_edge("generate", END)
graph.add_edge("no_answer", END)


app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"question": "what is 25 * 4 . what are the Machine learning approaches?"})
    print(result["answer"])

