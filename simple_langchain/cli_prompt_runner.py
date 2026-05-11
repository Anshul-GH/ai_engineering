from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def build_prompt_template(system_prompt: str) -> PromptTemplate:
    return PromptTemplate.from_template(
        "{system_prompt}\n\n"
        "User request:\n{user_input}\n\n"
        "Answer:"
    ).partial(system_prompt=system_prompt)


def make_approval_gate(enabled: bool):
    def approval_gate(prompt_value: StringPromptValue) -> StringPromptValue:
        if not enabled:
            return prompt_value

        final_prompt = prompt_value.to_string()
        print("\n===== APPROVAL GATE =====")
        print(final_prompt)
        print("=========================\n")

        while True:
            decision = input("Approve prompt? [y]es / [n]o / [e]dit: ").strip().lower()
            if decision in {"y", "yes", ""}:
                return prompt_value
            if decision in {"n", "no"}:
                raise RuntimeError("Prompt rejected by human reviewer.")
            if decision in {"e", "edit"}:
                print("Paste revised final prompt. End with Ctrl-D (macOS/Linux) or Ctrl-Z then Enter (Windows):")
                try:
                    edited = sys.stdin.read().strip()
                except KeyboardInterrupt:
                    raise RuntimeError("Prompt editing cancelled.")
                if not edited:
                    print("Edited prompt was empty. Keeping original prompt.")
                    return prompt_value
                return StringPromptValue(text=edited)
            print("Please enter y, n, or e.")

    return approval_gate


def make_ollama_caller(model_name: str, base_url: str):
    def call_ollama(prompt_value: StringPromptValue) -> str:
        prompt_str = prompt_value.to_string()
        payload = {
            "model": model_name,
            "prompt": prompt_str,
            "stream": False,
        }
        resp = httpx.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=120,
            trust_env=False,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    return call_ollama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Ollama-backed LangChain prompt pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["one-shot", "research", "rag"],
        default="one-shot",
        help="Run single-shot, research, or RAG flow.",
    )

    parser.add_argument(
        "--docs-path",
        help="Path to a text/markdown file or directory of docs for RAG mode.",
    )

    parser.add_argument(
        "--system-file",
        default="prompts/system.md",
        help="Path to system prompt file.",
    )

    parser.add_argument(
        "--model",
        default="llama3:latest",
        help="Ollama model name.",
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )

    parser.add_argument(
        "--user-input",
        help="User input text. If omitted, prompt interactively.",
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Require manual approval before model invocation.",
    )

    return parser.parse_args()

def load_docs(path: str) -> List[str]:
    p = Path(path)
    texts: List[str] = []
    if p.is_file():
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    elif p.is_dir():
        for f in p.rglob("*.txt"):
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))
        for f in p.rglob("*.md"):
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))
    else:
        raise FileNotFoundError(f"Docs path not found: {path}")
    return texts

def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def score_chunk(query: str, chunk: str) -> int:
    q_words = set(query.lower().split())
    c_words = set(chunk.lower().split())
    return len(q_words & c_words)

def run_rag(system_prompt: str, model: str, base_url: str, user_input: str, approve: bool, docs_path: str) -> str:
    if not docs_path:
        raise ValueError("--docs-path is required for rag mode.")

    # 1) Load and chunk docs
    raw_docs = load_docs(docs_path)
    all_chunks: List[str] = []
    for doc in raw_docs:
        all_chunks.extend(simple_chunk(doc))

    if not all_chunks:
        raise ValueError(f"No chunks loaded from {docs_path}")

    # 2) Score chunks by naive overlap
    scored = [(score_chunk(user_input, c), c) for c in all_chunks]
    scored = [item for item in scored if item[0] > 0] or scored
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:3]]  # take top 3

    context = "\n\n---\n\n".join(top_chunks)

    # 3) Build a RAG-style user input
    rag_user_input = (
        f"Use ONLY the context below to answer the question.\n"
        f"If the answer is not in the context, say you don't know.\n\n"
        f"Question:\n{user_input}\n\n"
        f"Context:\n{context}\n"
    )

    # 4) Call your existing pipeline
    return run_one_shot(system_prompt, model, base_url, rag_user_input, approve)

def run_research(system_prompt: str, model: str, base_url: str, user_input: str, approve: bool) -> str:
    # Shared state
    state = {
        "user_query": user_input,
        "search_query": "",
        "search_results": "",
        "final_answer": "",
    }

    # STEP 1: generate a search query
    search_system = system_prompt + "\n\nYou are a research planner. " \
        "Given the user query, produce a short, precise web search query.\n" \
        "Return ONLY the search query text, no explanation."

    search_query = run_one_shot(
        system_prompt=search_system,
        model=model,
        base_url=base_url,
        user_input=state["user_query"],
        approve=approve,
    )
    state["search_query"] = search_query.strip()

    # STEP 2: fake search results (stub; swap with a real tool later)
    state["search_results"] = (
        f"Result snippet 1 for '{state['search_query']}'...\n"
        f"Result snippet 2 for '{state['search_query']}'...\n"
        f"Result snippet 3 for '{state['search_query']}'...\n"
    )

    # STEP 3: summarize into final answer
    summarize_system = system_prompt + \
        "\n\nYou are a research summarizer. Use the provided search results " \
        "to answer the user query concisely."

    summarize_user_input = (
        f"User query:\n{state['user_query']}\n\n"
        f"Search results:\n{state['search_results']}\n"
    )

    final_answer = run_one_shot(
        system_prompt=summarize_system,
        model=model,
        base_url=base_url,
        user_input=summarize_user_input,
        approve=approve,
    )
    state["final_answer"] = final_answer.strip()
    return state["final_answer"]

def run_one_shot(system_prompt: str, model: str, base_url: str, user_input: str, approve: bool) -> str:
    prompt = build_prompt_template(system_prompt)
    approval_gate = RunnableLambda(make_approval_gate(approve))
    llm = RunnableLambda(make_ollama_caller(model, base_url))

    chain = RunnableSequence(
        prompt,
        approval_gate,
        llm,
        StrOutputParser(),
    )

    return chain.invoke({"user_input": user_input})

def main() -> None:
    args = parse_args()
    system_prompt = load_text(args.system_file)
    user_input = args.user_input or input("Enter user input: ").strip()

    if args.mode == "one-shot":
        result = run_one_shot(system_prompt, args.model, args.base_url, user_input, args.approve)
    elif args.mode == "research":
        result = run_research(system_prompt, args.model, args.base_url, user_input, args.approve)
    else:  # rag
        result = run_rag(system_prompt, args.model, args.base_url, user_input, args.approve, args.docs_path)

    print("\n===== MODEL RESPONSE =====")
    print(result)


if __name__ == "__main__":
    main()
