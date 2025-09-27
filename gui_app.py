"""
gui_app.py
简易 Tkinter GUI，用于在本地运行 RAG 检索 + 大模型问答流程
---------------------------------------------------------------
依赖：
  - Python 自带 Tkinter（大多数发行版默认包含）
  - 你的 d_search、call_olla 模块与其依赖
运行：
  python gui_app.py
"""
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import d_search
import call_olla

# ----------------- 核心逻辑函数（沿用你现有的实现） ----------------- #
def generate_answer(query: str, docs: list[str]) -> str:
    """
    将检索到的代码片段拼进 prompt，调用本地 Ollama 大模型生成回答。
    """
    context = "\n---\n".join(docs)
    prompt = (
        "你是一名经验丰富的代码高手。请根据用户描述与检索得到的相关五个代码片段，判断最贴合用户描述的那一条代码。"
        "如果没有合适的，也应勇敢指出并点明原因。"
        "如果上下文不足以回答，应如实说明。\n\n"
        "【用户描述】\n"
        f"{query}\n\n"
        "【检索结果】\n"
        f"{context}\n\n"
        "【回答】"
    )
    # 你可按需调整 model 名称
    return call_olla.call_ollama(prompt, model="qwen3:14b")


def retrieve_docs(model: str, query: str) -> list[str]:
    """
    根据所选模型检索文档，返回前 5 条代码片段列表。
    """
    if model == "gemini":
        return d_search.gemini_query(query, topk=5)
    else:
        return d_search.graphcodebert_query(query, topk=5)


# ----------------- Tkinter GUI ----------------- #
class RAGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("本地 RAG 检索问答 Demo")
        self.geometry("880x600")
        self._create_widgets()

    # 布局
    def _create_widgets(self):
        # -------- 顶部：模型选择 & 查询输入 -------- #
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="选择向量模型：").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="gemini")
        ttk.Radiobutton(top_frame, text="Gemini", variable=self.model_var,
                        value="gemini").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(top_frame, text="GraphCodeBERT", variable=self.model_var,
                        value="graphcodebert").pack(side=tk.LEFT)

        ttk.Label(top_frame, text="  查询：").pack(side=tk.LEFT, padx=(20, 0))
        self.query_entry = ttk.Entry(top_frame, width=50)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.query_entry.bind("<Return>", lambda _e: self._on_search())

        self.search_btn = ttk.Button(top_frame, text="检索并生成",
                                     command=self._on_search)
        self.search_btn.pack(side=tk.LEFT)

        # -------- 中下部：左右两个文本框 -------- #
        mid_frame = ttk.Frame(self, padding=10)
        mid_frame.pack(fill=tk.BOTH, expand=True)

        # 左：检索结果
        ttk.Label(mid_frame, text="🔎 检索结果（前 5 条）").pack(anchor=tk.W)
        self.docs_text = scrolledtext.ScrolledText(
            mid_frame, width=60, height=20, wrap=tk.WORD
        )
        self.docs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 右：生成回答
        ttk.Label(mid_frame, text="🤖 生成回答").pack(anchor=tk.W)
        self.answer_text = scrolledtext.ScrolledText(
            mid_frame, width=60, height=20, wrap=tk.WORD
        )
        self.answer_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

    # ---------- 事件回调 ---------- #
    def _on_search(self):
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showinfo("提示", "请输入查询后再执行~")
            return

        # 关闭按钮防抖，避免重复点击
        self.search_btn.config(state=tk.DISABLED)
        self.docs_text.delete("1.0", tk.END)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, "⏳ 正在检索并生成，请稍候...\n")

        # 子线程防止界面卡死
        threading.Thread(
            target=self._run_rag_pipeline,
            args=(self.model_var.get(), query),
            daemon=True,
        ).start()

    def _run_rag_pipeline(self, model: str, query: str):
        try:
            # ---------- Retrieval ----------
            docs = retrieve_docs(model, query)
            self.docs_text.insert(
                tk.END,
                "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(docs)]),
            )

            # ---------- Generation ----------
            answer = generate_answer(query, docs)
            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, answer)

        except Exception as e:
            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(
                tk.END, f"⚠️ 检索/生成出错：{e}"
            )
        finally:
            self.search_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    RAGApp().mainloop()
