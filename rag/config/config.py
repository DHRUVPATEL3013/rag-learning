from langchain_classic.storage import LocalFileStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(
model_name="BAAI/bge-small-en-v1.5"
)
model=HuggingFaceCrossEncoder(
   model_name="BAAI/bge-reranker-base"
)

parent_store = LocalFileStore("./parent_store")
