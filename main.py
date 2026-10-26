from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader=PyMuPDF4LLMLoader("data/sample_rag_file.pdf")
splitter=RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)
embeddings=HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents=loader.load()
chunks=splitter.split_documents(documents)

vectorstore=Chroma.from_documents(
  documents=chunks,
  embedding=embeddings
)

for i, chunk in enumerate(chunks):
    print(f"\n{'=' * 50}")
    print(f"CHUNK {i}")
    print(f"Length: {len(chunk.page_content)}")
    print(f"Page: {chunk.metadata.get('page')}")
    print(f"{'=' * 50}")
    print(chunk.page_content)

results=vectorstore.similarity_search(
  "be communicated within the organization",
  k=3
)

for result in results:
  print(result.page_content)
  print(result.metadata)
  print("-" * 50)




