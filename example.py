from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_chroma import Chroma

loader=PyMuPDF4LLMLoader("data/rag-sample.pdf")

docs=loader.load()

docs = [
    d for d in docs
    if "REFERENCES" not in d.page_content[:200]
    and "arXiv preprint" not in d.page_content[:100]
]

md_splitter=MarkdownHeaderTextSplitter(
  headers_to_split_on=[
    ("#","h1"),
    ("##","h2"),
    ("###","h3"),
    ("####","h4"),
  ],
  strip_headers=False
)

sectioned=[]

for doc in docs:
  for s in md_splitter.split_text(doc.page_content):
    s.metadata.update(doc.metadata)
    sectioned.append(s)



def is_table(text:str):
  return text.count("|") > 20 and "\n|" in text

fallback=RecursiveCharacterTextSplitter(

  chunk_size=2000,
  chunk_overlap=200

)


final_chunks=[]

for c in sectioned:
  if is_table(c.page_content) or len(c.page_content) <=2000:
    final_chunks.append(c)
  else:
    for sc in fallback.split_documents([c]):
      final_chunks.append(sc)

final_chunks=[c for c in final_chunks if len(c.page_content.strip()) > 150]

for c in final_chunks:
  header=">".join(filter (
    None,[
      c.metadata.get("h1"),
      c.metadata.get("h2"),
      c.metadata.get("h3"),
      c.metadata.get("h4"),
    ]
  ))

  if header and not c.page_content.startswith("["):
    c.page_content=f"[{header}]\n{c.page_content}"


# with open("output.txt", "w") as f:

#     for i, chunk in enumerate(final_chunks):

#         f.write(f"\n\n========== CHUNK {i + 1} ==========\n\n")

#         f.write("PAGE CONTENT:\n")
#         f.write(chunk.page_content)

#         f.write("\n\nMETADATA:\n")
#         f.write(str(chunk.metadata))

#         f.write("\n\n")
#         f.write("*" * 80)
#         f.write("\n")



embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5" 
)

vectorstore=Chroma.from_documents(
  final_chunks,
  embedding=embeddings
)

retriver=vectorstore.as_retriever(
    search_type="mmr",
       search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)


def format_docs(docs):
  return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a research paper.

Rules:
1. Use ONLY the information in the context below.
2. If the context does not contain the answer, respond exactly:
   "I don't know based on the provided context."
3. If the context contains only part of the answer, answer with what you can
   and explicitly state what is missing.
4. Cite the section heading(s) (shown in square brackets, e.g. [III. RETRIEVAL])
   that your answer comes from.
5. Be concise — 2 to 5 sentences unless the question requires a list.
6. Do not use prior knowledge. Do not speculate.

Context:
{context}

Question: {question}

Answer:"""
)

llm=ChatOllama(
  model="qwen3:4b"
)


rag_chain=(
  {"context":retriver | format_docs , "question":RunnablePassthrough()}
  |prompt
  |llm
  |StrOutputParser()
)

questions = [
    # ---------- Category 1: Factual lookup ----------
    "What is the 'Retrieve-Read' framework?",

    # ---------- Category 6: Comparative ----------
    "What are the differences between iterative, recursive, and adaptive retrieval?",

    # ---------- Category 7: Negative tests (should say 'I don't know') ----------
    "Who is the CEO of OpenAI according to this paper?",
]

# answers=rag_chain.batch(questions,config={"max_concurrency": 1})
answer=rag_chain.invoke("what is rag?")
print(answer)



# with open("rag_answers.txt", "w") as f:
#     for q,a in zip(questions,answers):
#             f.write(f"\nQ: {q}\nA: {a}\n")
#             f.write("*"* 100)