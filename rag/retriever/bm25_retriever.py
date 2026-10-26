from rag.retriever.dense_retriever import vectorstore
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


all_data=vectorstore.get()
all_texts=all_data["documents"]
all_metadatas=all_data["metadatas"]


all_documents=[
  Document(page_content=text,metadata=meta)
  for text ,meta in zip(all_texts,all_metadatas)
]

bm25_retriver=BM25Retriever.from_documents(all_documents)
bm25_retriver.k=20


# print(all_data.keys())
