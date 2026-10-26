import re 
from rag.splitters.splitters import md_splitter,fallback
from pypdf import PdfReader


def clean_wiki_md(text):
    # Remove [edit] markers
    text = re.sub(r"\[edit\]\n?", "", text)
    # Remove footnote markers
    text = re.sub(r"<sup>\[\d+\]</sup>", "", text)
    # Remove [citation needed]
    text = re.sub(r"\[.*?citation needed.*?\]", "", text)
    # Remove sidebar tables (series boxes)
    text = re.sub(r"\| Part of a series on.*?\n\n", "", text, flags=re.DOTALL)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    text=re.sub(r"<sup>.*?</sup>", "", text)
    return text





def get_header_and_apply_fallback_splitter(chunks):
  final_chunks=[]
  for c in chunks:
    if len(c.page_content) < 3000:
      final_chunks.append(c)

    else:
      header=">".join(filter(
        None,
        [
            c.metadata.get("h1"),c.metadata.get("h2"),
            c.metadata.get("h3"),c.metadata.get("h4"),
        ]
      ))
      for sc in fallback.split_documents([c]):
        if header:
          sc.page_content=f"[{header} \n {sc.page_content}]"
        final_chunks.append(sc)
  return final_chunks





def is_noise_section(breadcrumb):
    noise = ["References", "See also", "Further reading", "External links", "Sources"]
    return any(n in breadcrumb for n in noise)



# Merge tiny forward
def merge_tiny_chunks(chunks):
    
  merged = []
  for c in chunks:
      if merged and len(merged[-1].page_content) < 300:
          merged[-1].page_content += "\n\n" + c.page_content
      else:
          merged.append(c)
  return merged
  





def process_document(filepath:str):
    if filepath.lower().endswith(".pdf"): 
      reader = PdfReader(filepath) 
      content = "\n".join( page.extract_text() or "" for page in reader.pages )
    else:
      with open(filepath) as f:
        content=f.read()

    content = clean_wiki_md(content)
    filename=filepath.split("/")[-1].split(".")[0]
    chunks=md_splitter.split_text(content)

    final_chunks=get_header_and_apply_fallback_splitter(chunks)

    for i,c in enumerate(final_chunks):
      breadcrumb = " > ".join(filter(None, [
          c.metadata.get("h1"),
          c.metadata.get("h2"),
          c.metadata.get("h3"),
          c.metadata.get("h4"),
      ]))
      c.metadata["breadcrumb"] = breadcrumb
      c.metadata["source"] = filename
    final_chunks = [c for c in final_chunks if not is_noise_section(c.metadata.get("breadcrumb", ""))]


    merged_chunks=merge_tiny_chunks(final_chunks)

    #drop tiny chunks
    final_chunks = [c for c in merged_chunks if len(c.page_content.strip()) > 200]

    with open("rag/chunks/hologram-chunk.txt","w") as f:
       for i,chunk in enumerate(final_chunks):
          f.write(f"Chunk {i+1}\n\n")
          f.write(f"{chunk.page_content}\n\n")
          f.write(f"{chunk.metadata}\n")
          f.write("*"* 100)
          f.write("\n\n")

    return final_chunks



    





 





