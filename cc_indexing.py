import chromadb
import b_emb


BATCH_SIZE  = 8



chromadb_client = chromadb.PersistentClient("./chroma_data")

def create_db_gemini(
    code_texts: list[str],
    query_texts: list[str],
    batch_size: int = BATCH_SIZE
) -> None:
    col = chromadb_client.get_or_create_collection("gemini_base")
    for i in range(0, len(code_texts), batch_size):
        batch_texts = code_texts[i : i + batch_size]

        batch_vecs = b_emb.gemini_embed(batch_texts, store=True)
        ids = query_texts[i: i+ batch_size]
        col.upsert(
            ids=ids,
            documents=batch_texts,
            embeddings=batch_vecs
        )
        print(f"[info] 已写入 {len(batch_texts)} 条，进度 {i + len(batch_texts)}/{len(code_texts)}")

def create_db_graphcodebert(
    code_texts: list[str],
    query_texts: list[str],
    batch_size: int = BATCH_SIZE
) -> None:
    col = chromadb_client.get_or_create_collection("graphcodebert_base")
    for i in range(0, len(code_texts), batch_size):
        batch_texts = code_texts[i : i + batch_size]
        batch_vecs = b_emb.graphcodebert_embed(batch_texts)
        ids = query_texts[i: i+ batch_size]
        col.upsert(
            ids=ids,
            documents=batch_texts,
            embeddings=batch_vecs
        )
        print(f"[info] 已写入 {len(batch_texts)} 条，进度 {i + len(batch_texts)}/{len(code_texts)}")






