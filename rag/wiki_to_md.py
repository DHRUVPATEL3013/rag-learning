import trafilatura

def wiki_to_md(url:str):
  # url="https://en.wikipedia.org/wiki/Machine_learning"
  downloaded=trafilatura.fetch_url(url)

  text=trafilatura.extract(
    downloaded,
    include_tables=True,
    include_formatting=True,
    output_format="markdown"
  )
  file_name=url.split("/")
  file_name=f"{file_name[-1]}.md"

  with open(f"rag/rag-data/{file_name}","w",encoding="utf-8") as f:
    f.write(text)

wiki_to_md("https://en.wikipedia.org/wiki/Holography")