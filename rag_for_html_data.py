from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from markdownify import markdownify
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------- 1. Load HTML ----------
with open("data/ufc-html-doc.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")


# ---------- 2. Extract tables separately ----------
def html_table_to_markdown(table_tag) -> str:
    """Convert a <table> tag to a Markdown table."""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            text = cell.get_text(strip=True).replace("|", "\\|")
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")

    if not rows:
        return ""

    # Add separator after header row
    header = rows[0]
    num_cols = header.count("|") - 1
    separator = "|" + "---|" * num_cols
    return header + "\n" + separator + "\n" + "\n".join(rows[1:])


# Grab all tables, then remove them from the soup
tables = soup.find_all("table")
table_chunks = []

for i, table in enumerate(tables):
    # Find nearest heading above the table for context
    heading = table.find_previous(["h1", "h2", "h3"])
    heading_text = heading.get_text(strip=True) if heading else "Table"

    md = html_table_to_markdown(table)
    if not md.strip():
        continue
    
    content = f"[{heading_text}]\n{md}"
    table_chunks.append(Document(
        page_content=content,
        metadata={"type": "table", "table_index": i, "heading": heading_text},
    ))

# Remove tables from soup so they don't get re-chunked as prose
for t in tables:
    t.decompose()


# ---------- 3. Convert remaining HTML to Markdown ----------
prose_html = str(soup)

# Keep only <main> content (drop header/footer/aside if desired)
main_tag = soup.find("main")
if main_tag:
    prose_html = str(main_tag)
else:
    prose_html = str(soup)

prose_md = markdownify(prose_html, heading_style="ATX")


# ---------- 4. Split prose by headings ----------
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ],
    strip_headers=False,
)

prose_chunks = md_splitter.split_text(prose_md)


# ---------- 5. Split oversized prose ----------
fallback = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

final_prose = []
for c in prose_chunks:
    if len(c.page_content) <= 1500:
        final_prose.append(c)
    else:
        final_prose.extend(fallback.split_documents([c]))


# ---------- 6. Merge tiny chunks forward ----------
merged = []
for c in final_prose:
    if merged and len(merged[-1].page_content) < 300:
        merged[-1].page_content += "\n\n" + c.page_content
    else:
        merged.append(c)
final_prose = merged


# ---------- 7. Drop tiny / empty chunks ----------
final_prose = [c for c in final_prose if len(c.page_content.strip()) > 100]


# ---------- 8. Attach breadcrumbs (metadata only) ----------
for c in final_prose:
    header = " > ".join(filter(None, [
        c.metadata.get("h1"),
        c.metadata.get("h2"),
        c.metadata.get("h3"),
        c.metadata.get("h4"),
    ]))
    c.metadata["breadcrumb"] = header
    c.metadata["type"] = "prose"


# ---------- 9. Combine prose + table chunks ----------
final_chunks = final_prose + table_chunks

print(f"Prose chunks: {len(final_prose)}")
print(f"Table chunks: {len(table_chunks)}")
print(f"Total: {len(final_chunks)}")


# ---------- 10. Index ----------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma.from_documents(
    documents=final_chunks,
    embedding=embeddings,
)


# ---------- 11. Test retrieval ----------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20},
)


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
  {"context":retriever | format_docs , "question":RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)



answer=rag_chain.invoke("What are the UFC weight divisions?")

print(answer)


