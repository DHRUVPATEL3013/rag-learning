from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter


md_splitter=MarkdownHeaderTextSplitter(
    headers_to_split_on=[
    ("#","h1"),
    ("##","h2"),
    ("###","h3"),
    ("####","h4"),
  ],
  strip_headers=False           # keep the headers in the chunk
)

fallback=RecursiveCharacterTextSplitter(
  chunk_size=3000,
  chunk_overlap=250
)

child_splitter=RecursiveCharacterTextSplitter(
  chunk_size=400
)


