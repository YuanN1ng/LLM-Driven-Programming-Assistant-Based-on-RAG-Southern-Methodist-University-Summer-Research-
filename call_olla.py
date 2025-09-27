import requests


import requests
import json
import re

def call_ollama(prompt: str, model: str = "qwen3:14b") -> str:
    """
    :param prompt: 输入文本（可为问题、摘要请求等）
    :param model: 模型名称，默认 llama3
    :return: 模型返回的文本
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print("请求出错：", e)
        return ""




