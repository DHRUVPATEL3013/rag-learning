from langchain_chroma import Chroma
from rag.config.config import embeddings,parent_store
from rag.splitters.splitters import child_splitter
from rag.process_document import process_document
import pickle


import os

def store_vectore(chunks_to_store):
    if os.path.exists("./chroma-db"):
        vectorstore = Chroma(
            persist_directory="./chroma-db",
            embedding_function=embeddings,
        )
        vectorstore.add_documents(chunks_to_store)
    else:
      
        vectorstore = Chroma.from_documents(
            documents=chunks_to_store,
            embedding=embeddings,
            persist_directory="./chroma-db",
        )
    return vectorstore



parent_chunks=process_document("./rag/rag-data/employee_attendance_policy.pdf")

# children=[]
# for parent_chunk in parent_chunks:
#   parent_id=parent_chunk.metadata["parent_id"]
  
#   parent_store.mset([(parent_id,pickle.dumps(parent_chunk))])
#   for child in child_splitter.split_documents([parent_chunk]):
#     child.metadata["parent_id"]=parent_chunk.metadata["parent_id"]
#     children.append(child)

if __name__=="__main__":
    store_vectore(parent_chunks) 












