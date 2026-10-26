
from langchain_chroma import Chroma
from rag.config.config import embeddings,model



vectorstore=Chroma(
  persist_directory="./chroma-db",
  embedding_function=embeddings
)
dense_retriver=vectorstore.as_retriever(
  search_type="mmr",
  search_kwargs={"k":7,"fetch_k":20},
)





