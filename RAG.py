import d_search
import call_olla






def choose_model() -> str:
    """让用户选择检索模型；返回 'gemini' 或 'graphcodebert'。"""
    while True:
        print("\n请选择向量模型：")
        print("[1] Gemini")
        print("[2] GraphCodeBERT")
        choice = input("输入编号并回车: ").strip()
        if choice == "1":
            return "gemini"
        elif choice == "2":
            return "graphcodebert"
        else:
            print("❌ 无效输入，请重新选择（1 或 2）。")


def generate_answer(query: str, docs: list[str]) -> str:
    """
    把检索到的代码片段拼进 prompt，调用本地 Ollama 大模型生成回答。
    """
    context = "\n---\n".join(docs)
    prompt = (
        "你接下来只需要关注用户最新输入，对之前的对话内容一律忽略。你是一名经验丰富的代码高手。请根据用户的自然语言描述与检索得到的相关五个代码片段，首先在这五条代码片段中指出最贴合用户需求的那一条代码。"
        "如果检索出来的五条代码中没有合适的，也说明情况并点明原因，同时给出检索出来的五条代码之外的合适的代码。"
        "如果上下文不足以回答，应如实说明。\n\n"
        "【用户自然语言描述】\n"
        f"{query}\n\n"
        "【检索出的五个代码片段】\n"
        f"{context}\n\n"
        "【回答】"
    )
    return call_olla.call_ollama(prompt, model="qwen3:14b")


def run_query_loop():
    model = choose_model()
    print(f"\n✅ 已切换到 {model} 模型。")
    print("输入自然语言开始 RAG；输入 'switch' 可切换模型；输入 'quit' 退出。")

    while True:
        query = input("\n>>> ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("👋 再见！")
            break
        if query.lower() == "switch":
            model = choose_model()
            print(f"\n✅ 已切换到 {model} 模型。")
            continue

        try:
            # ---------- Retrieval ----------
            if model == "gemini":
                docs = d_search.gemini_query(query)
            else:
                docs = d_search.graphcodebert_query(query)

            print("\n🔎 检索结果（前 5 条）：")
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc[:120]}{'...' if len(doc) > 120 else ''}")

            # ---------- Generation ----------
            answer = generate_answer(query, docs)
            print("\n🤖 生成回答：")
            print(answer)

        except Exception as e:
            print("⚠️ 检索/生成时出错：", e)


if __name__ == "__main__":
    run_query_loop()
