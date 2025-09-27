# delete_graphcodebert_collection.py
import chromadb
client = chromadb.PersistentClient("./chroma.db")
try:
    client.delete_collection("graphcodebert_base")
    print("✅ 旧 graphcodebert_base 已删除")
except chromadb.errors.CollectionNotFoundError:
    print("⚠️ 没找到旧集合，直接跳过")
