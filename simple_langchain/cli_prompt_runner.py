from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        "--system-file",
        default="output/prompts/system.md",
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


def main() -> None:
    args = parse_args()
    system_prompt = load_text(args.system_file)
    user_input = args.user_input or input("Enter user input: ").strip()

    prompt = build_prompt_template(system_prompt)
    approval_gate = RunnableLambda(make_approval_gate(args.approve))
    llm = RunnableLambda(make_ollama_caller(args.model, args.base_url))

    chain = RunnableSequence(
        prompt,
        approval_gate,
        llm,
        StrOutputParser(),
    )

    result = chain.invoke({"user_input": user_input})
    print("\n===== MODEL RESPONSE =====")
    print(result)


if __name__ == "__main__":
    main()
