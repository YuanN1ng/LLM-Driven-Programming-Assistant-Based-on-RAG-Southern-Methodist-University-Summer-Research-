import chromadb
import b_emb

_client = chromadb.PersistentClient("./chroma_data")

def query(model, question, topk):
    if model=="gemini":
        col = _client.get_or_create_collection("gemini_base")
        q_vec = b_emb.gemini_embed(question, store=False)[0]
        result = col.query(
            query_embeddings=[q_vec],
            n_results=topk,
            include=["documents"]
        )
        documents = result["documents"][0]
        ids = result["ids"][0]
    else:
        col = _client.get_or_create_collection("graphcodebert_base")
        q_vec = b_emb.graphcodebert_embed([question])[0]
        result = col.query(
            query_embeddings=[q_vec],
            n_results=topk,
            include=["documents"]
        )
        documents = result["documents"][0]
        ids = result["ids"][0]
    return ids, documents

def gemini_query(question: str, topk: int = 5):
    col = _client.get_or_create_collection("gemini_base")
    q_vec = b_emb.gemini_embed(question, store=False)[0]
    result = col.query(
        query_embeddings=[q_vec],
        n_results=topk,

    )
    documents = result["documents"][0]

    return documents


def graphcodebert_query(question: str, topk: int = 5):
    col = _client.get_or_create_collection("graphcodebert_base")
    q_vec = b_emb.graphcodebert_embed([question])[0]
    result = col.query(
        query_embeddings=[q_vec],
        n_results=topk,

    )
    documents = result["documents"][0]

    return documents

