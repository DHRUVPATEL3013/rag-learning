from langchain_classic.retrievers import EnsembleRetriever
from rag.retriever.dense_retriever import dense_retriver
from rag.retriever.bm25_retriever import bm25_retriver
from rag.config.config import model
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever


ensemble_retriver=EnsembleRetriever(
  retrievers=[dense_retriver,bm25_retriver],
  weights=[0.5,0.5]
)

reranker=CrossEncoderReranker(
  model=model,
  top_n=5
)

final_retriver=ContextualCompressionRetriever(
  base_compressor=reranker,
  base_retriever=ensemble_retriver
)

