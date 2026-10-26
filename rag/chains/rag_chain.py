

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.retriever.dense_retriever import dense_retriver
from rag.llm import llm,format_docs
from rag.prompts.prompts import prompt




rag_chain=(
  {"context":dense_retriver | format_docs ,"question":RunnablePassthrough()}
  |prompt
  |llm
  |StrOutputParser()
)








