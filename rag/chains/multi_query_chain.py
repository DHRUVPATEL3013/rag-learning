from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm import llm
from retriever.dense_retriever import dense_retriver
from prompts.prompts import multi_query_prompt


query_llm_chain=multi_query_prompt | llm | StrOutputParser()


multi_query_retriever=MultiQueryRetriever.from_llm(
retriever=dense_retriver,
llm=llm,
prompt=multi_query_prompt
)