# app.py

import httpx
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompt_values import StringPromptValue  # for type hints only


def call_ollama(prompt_value: StringPromptValue) -> str:
    """Call local Ollama; convert PromptValue -> str before sending."""
    prompt_str = prompt_value.to_string()  # <— key fix

    payload = {
        "model": "llama3:latest",
        "prompt": prompt_str,
        "stream": False,
    }

    resp = httpx.post(
        "http://127.0.0.1:11434/api/generate",
        json=payload,
        timeout=60,
        trust_env=False,  # bypass any proxies
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def simple_llm_demo():
    prompt = PromptTemplate.from_template(
        "You are a concise Python tutor.\n"
        "Explain what a Python virtual environment is in 2 sentences."
    )

    llm = RunnableLambda(call_ollama)

    # prompt -> StringPromptValue -> call_ollama -> str
    chain = RunnableSequence(prompt, llm, StrOutputParser())

    print(chain.invoke({}))


if __name__ == "__main__":
    simple_llm_demo()
