import chromadb
import b_emb
from pathlib import Path
#导入向量
CODE_FILE   = r"data\codes_top50.txt"   # 生成的 txt
BATCH_SIZE  = 8                            # 可根据显存调大
from pathlib import Path


chromadb_client= chromadb.PersistentClient("./chroma.db")



def create_db_gemini(txt_path: str | Path,
                    separator: str = "# === END_OF_CODE ===",
                    batch_size: int = BATCH_SIZE) -> None:
    chromadb_collection = chromadb_client.get_or_create_collection("gemini_base")


    codes = Path(txt_path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in codes.split(separator) if b.strip()]
    print(f"[info] 文件切出 {len(blocks)} 段代码")


    for start in range(0, len(blocks), batch_size):

        batch = blocks[start:start + batch_size]
        vecs = b_emb.gemini_embed(batch, store=True)
        ids = [f"{start + i}" for i in range(len(batch))]
        chromadb_collection.upsert(ids=ids,
                                                                           documents = batch,
                                       embeddings = vecs)
        print(f"[info] 已写入 {len(batch)} 条，进度 {start + len(batch)}/{len(blocks)}")



def create_db_graphcodebert(txt_path: str | Path,
                    separator: str = "# === END_OF_CODE ===",
                    batch_size: int = BATCH_SIZE) -> None:
    chromadb_collection = chromadb_client.get_or_create_collection("graphcodebert_base")


    codes = Path(txt_path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in codes.split(separator) if b.strip()]
    print(f"[info] 文件切出 {len(blocks)} 段代码")


    for start in range(0, len(blocks), batch_size):
        batch = blocks[start:start + batch_size]
        vecs = b_emb.graphcodebert_embed(batch)
        ids = [f"{start + i}" for i in range(len(batch))]
        chromadb_collection.upsert(ids=ids,
                                   documents = batch,
                                   embeddings = vecs)
        print(f"[info] 已写入 {len(batch)} 条，进度 {start + len(batch)}/{len(blocks)}")







