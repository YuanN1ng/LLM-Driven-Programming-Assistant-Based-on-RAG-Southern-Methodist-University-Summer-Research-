import chromadb, d_search, call_olla
from collections import Counter

client = chromadb.PersistentClient(path="./chroma_data")

def retrieve(model, query, k=5):
    ids, docs = d_search.query(model, query, topk=k)
    return ids, docs

def rerank(query, docs):
    prompt = (
        "你是一名经验丰富的代码高手。请根据用户描述与检索得到的相关五个代码片段，判断最贴合用户描述的那一个代码片段并返回其标号，务必记住除单个标号外不要回复任何其他内容。"
        "例如，当你认为第三个代码片段最合适时，你只需回答'3'。"
        "如果上下文不足以回答，应如实说明。\n\n"
        f"{query}\n\n【检索结果】\n" + "\n---\n".join(docs) + "\n\n【回答】"
    )
    ans = call_olla.call_ollama(prompt, model="qwen3:14b").strip()
    return int(ans) if ans.isdigit() else None

def evaluate(model):
    col = client.get_collection(f"{model}_base")
    data = col.get(include=["documents"])
    metrics = Counter()
    for q_id, q_doc in zip(data["ids"], data["documents"]):
        ids, docs = retrieve(model, q_doc, k=5)
        hit_retrieval = (ids[0] == q_id)
        rank = rerank(q_doc, docs)
        hit_rerank = rank is not None and rank-1 < len(ids) and ids[rank-1] == q_id
        metrics["retr_hit"] += hit_retrieval
        metrics["rerank_hit"] += hit_rerank
    n = len(data["ids"])
    print(f"[{model}] Retrieval Hit@1: {metrics['retr_hit']/n:.2%} | "
          f"Rerank Acc: {metrics['rerank_hit']/n:.2%}")

if __name__ == "__main__":
    model="graphcodebert"
    evaluate(model)
'''    for model in ("gemini", "graphcodebert"):
        evaluate(model)
        '''