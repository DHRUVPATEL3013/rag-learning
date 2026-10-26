from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama



llm=ChatOllama(
  model="qwen3:4b",
  temperature=0
)

def format_docs(docs:list):
  return "\n\n".join(d.page_content for d in docs)