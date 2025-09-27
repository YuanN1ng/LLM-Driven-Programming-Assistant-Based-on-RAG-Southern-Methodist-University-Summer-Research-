from google import genai
from google.genai import types
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
from typing import List
# ---------------- 配置 ----------------
CODE_FILE   = r"data\codes_top50.txt"   # 生成的 txt
BATCH_SIZE  = 8                            # 可根据显存调大
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base",
                                  add_pooling_layer=False).to(DEVICE).eval()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "❌ 未检测到环境变量 GEMINI_API_KEY，请先在系统或 Run/Debug 配置中设置。")


@torch.no_grad()
def gemini_embed(texts: list[str] | str, *, store: bool) -> list[list[float]]:
    """Gemini embedding → 返回 list[list[float]] 以便直接写入 Chroma"""
    if isinstance(texts, str):
        texts = [texts]
    client = genai.Client(api_key=API_KEY)  # 用环境变量
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config={"task_type": "RETRIEVAL_DOCUMENT" if store else "CODE_RETRIEVAL_QUERY"},
    )
    # SDK 返回 EmbedContentResponse，需取 .embedding.values
    return [e.values for e in resp.embeddings]




@torch.no_grad()
def graphcodebert_embed(code_list: List[str]) -> List[List[float]]:
    """
    输入: 多条代码字符串
    输出: 每条代码一个 768 维向量 (List[List[float]])
    """
    inp = tok(code_list,
               return_tensors="pt",
               padding=True,
               truncation=True,
               max_length=512).to(DEVICE)

    cls_vec = model(**inp).last_hidden_state[:, 0, :]   # (batch, 768)
    cls_vec = F.normalize(cls_vec, dim=1)                # 按行归一化
    return cls_vec.cpu().tolist()                        # List[List[float]]



