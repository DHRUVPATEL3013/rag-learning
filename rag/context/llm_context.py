def reorder_docs(docs):
  if len(docs) <=2:
    return docs

  first_half=docs[::2]
  second_half=docs[1::2]

  return first_half + second_half[::-1]

def fit_to_budget(docs,max_tokens=22000):
  docs_sorted=sorted(docs,key=lambda d : d.metadata.get("score",0),reverse=True)

  kept=[]
  total=0

  for d in docs_sorted:
    tokens=len(d.page_content) // 4
    if (total+tokens) > max_tokens:
      continue          # skip this one ,try next
    kept.append(d)
    total +=tokens
  return sorted(kept, key=lambda d: d.metadata.get("score", 0), reverse=True)
