import json
from datetime import datetime

def log_query(question, answer, grounded, relevant, latency_ms):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer[:200],      # truncate
        "grounded": grounded,
        "relevant": relevant,
        # "sources": list(set(sources)),
        # "chunks_count": len(sources),
        "latency_ms": latency_ms,
        "model": "qwen3:4b",
    }

    with open("rag_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")