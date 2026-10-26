from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

pdf_loader=PyMuPDF4LLMLoader("data/sample-rag-file.pdf")

documents=pdf_loader.load()    #if there is 20 pages then it will load 20 documntes .1 page = 1 documnet

md_splitter=MarkdownHeaderTextSplitter(
  headers_to_split_on=[
    ("#","h1"),
    ("##","h2"),
    ("###","h3"),
    ("####","h4"),
  ],
  strip_headers=False           # keep the headers in the chunk
)


sectioned=[]

for doc in documents:
  for s in md_splitter.split_text(doc.page_content):
    s.metadata.update(doc.metadata)
    sectioned.append(s)
def extract_markdown_table_rows(text):
    """Return list of rows if text contains a Markdown table, else None."""
    lines = text.split("\n")
    pipe_lines = [l for l in lines if l.strip().startswith("|")]

    if len(pipe_lines) < 3:
        return None

    # Build rows: consecutive pipe-lines become one row;
    # non-pipe lines are kept as separate chunks.
    rows = []
    pending_text = []
    for line in lines:
        if line.strip().startswith("|"):
            if pending_text:
                rows.append(("text", "\n".join(pending_text)))
                pending_text = []
            rows.append(("row", line))
        else:
            pending_text.append(line)
    if pending_text:
        rows.append(("text", "\n".join(pending_text)))

    return rows


def split_tables(chunks):
    """Split table chunks into row chunks; keep non-table chunks as-is."""
    expanded = []
    for c in chunks:
        header = " > ".join(filter(None, [
            c.metadata.get("h1"), c.metadata.get("h2"),
            c.metadata.get("h3"), c.metadata.get("h4"),
        ]))

        rows = extract_markdown_table_rows(c.page_content)
        if not rows:
            expanded.append(c)
            continue

        for kind, content in rows:
            if kind == "row":
                # Each table row becomes its own chunk with heading prefix
                page_content = f"[{header}]\n{content}" if header else content
                expanded.append(Document(
                    page_content=page_content,
                    metadata={**c.metadata, "type": "table_row"},
                ))
            else:
                # Non-table text around the table
                page_content = f"[{header}]\n{content}" if header else content
                expanded.append(Document(
                    page_content=page_content,
                    metadata=dict(c.metadata),
                ))

    return expanded


sectioned = split_tables(sectioned)
fallback=RecursiveCharacterTextSplitter(
  chunk_size=2500,
  chunk_overlap=200,
  separators=["\n\n","\n",". "," ",""],

)

final_chunks=[]

for c in sectioned:
  if len(c.page_content) <= 2500:
    final_chunks.append(c)
  else:
    header=">".join(
      filter(
        None,[
          c.metadata.get("h1"),c.metadata.get("h2"),
          c.metadata.get("h3"),c.metadata.get("h4"),
        ]
      )
    )
    for sc in fallback.split_documents([c]):
      if header and not sc.page_content.startswith("["):
        sc.page_content =  f"[{header}]\n{sc.page_content}"
      final_chunks.append(sc)

final_chunks=[c for c in final_chunks if len(c.page_content.strip()) > 100]

for c in final_chunks:
    header = " > ".join(filter(None, [
        c.metadata.get("h1"), c.metadata.get("h2"),
        c.metadata.get("h3"), c.metadata.get("h4"),
    ]))
    if header and not c.page_content.startswith("["):
        c.page_content = f"[{header}]\n{c.page_content}"

embeddings=HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_documents(
  documents=final_chunks,
  embedding=embeddings
)




# query="what is the information security policy ? "
# results=vectorstore.similarity_search(query,k=3)

# for i,r in enumerate(results):
#   print(f"result {i+1}")
#   print(f"Metadata: {r.metadata}")
#   print(f"Content:\n{r.page_content[:500]}...")

# results = vectorstore.similarity_search("What is control 5.15?", k=5)

# for i, r in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(f"Length: {len(r.page_content)}")
#     print(f"First 200 chars:\n{r.page_content[:200]}")
#     print(f"Contains '5.15'? {'5.15' in r.page_content}")


retriver=vectorstore.as_retriever(search_kwargs={"k":3})

def format_docs(docs):
  return "\n\n".join(d.page_content for d in docs)

prompt=ChatPromptTemplate.from_template(
  """
Answer the question using only the context below.
if the answer not in the context.say "I don't know" 


Context:
{context}

Question:{question}

Answer:
"""
)


llm=ChatOllama(
  model="qwen3:4b"
)

rag_chain=(
  {"context":retriver | format_docs , "question":RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)


question="tell me about modular rag"

answer=rag_chain.invoke(question)
print(answer)









