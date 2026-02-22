"""Prompt and LLM chain utilities for grounded QA."""

from __future__ import annotations

from collections import OrderedDict
import os
import re
from typing import Iterable

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

import config


def _format_docs(docs: Iterable) -> tuple[str, dict[str, str]]:
    snippets = []
    sources: dict[str, str] = OrderedDict()

    for i, doc in enumerate(docs, start=1):
        citation = f"[{i}]"
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_str = f", page {page + 1}" if isinstance(page, int) else ""
        sources[citation] = f"{src}{page_str}"
        snippets.append(f"{citation} {doc.page_content}")

    return "\n\n".join(snippets), sources


def _format_history(chat_history: list | None, max_messages: int = 6) -> str:
    if not chat_history:
        return "None"

    trimmed = chat_history[-max_messages:]
    lines = []
    for msg in trimmed:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role.title()}: {content}")
            continue

        if isinstance(msg, (tuple, list)) and len(msg) == 2:
            user_msg = str(msg[0] or "").strip()
            assistant_msg = str(msg[1] or "").strip()
            if user_msg:
                lines.append(f"User: {user_msg}")
            if assistant_msg:
                lines.append(f"Assistant: {assistant_msg}")

    return "\n".join(lines) if lines else "None"


def _build_llm() -> HuggingFacePipeline:
    token = os.getenv("HF_TOKEN")
    model_kwargs = {"token": token} if token else {}

    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL, **model_kwargs)

    if torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            dtype=dtype,
            **model_kwargs,
        )
    except TypeError:
        # Backward compatibility with older transformers.
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=dtype,
            **model_kwargs,
        )
    model.to(device)

    gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=config.TEMPERATURE > 0,
        temperature=config.TEMPERATURE,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


LLM = _build_llm()


def _contains_math(text: str) -> bool:
    math_markers = [r"\\frac", r"\\Theta", r"\\Omega", r"\\log", r"\^", r"_", "="]
    return any(re.search(marker, text) for marker in math_markers)


def _has_math_format_issues(text: str) -> bool:
    # Unbalanced display math delimiters.
    if text.count("$$") % 2 != 0:
        return True

    # Unbalanced inline math delimiters after removing display blocks.
    without_display = text.replace("$$", "")
    if without_display.count("$") % 2 != 0:
        return True

    # Common malformed pseudo-math wrappers.
    if re.search(r"\[[^\]\n]*(?:\\frac|\\Theta|\\Omega|\\log|=|\^|_)[^\]\n]*\]", text):
        return True
    if re.search(r"\([^\)\n]*(?:\\frac|\\Theta|\\Omega|\\log|=|\^|_)[^\)\n]*\)", text):
        return True

    return False


def _repair_math_format(answer: str) -> str:
    repair_prompt = PromptTemplate.from_template(
        "You are a strict Markdown+LaTeX formatter.\n"
        "Fix only formatting issues in the answer below.\n"
        "Do not change mathematical meaning, claims, or citations.\n"
        "Rules:\n"
        "- Keep output in Markdown.\n"
        "- Inline math uses $...$.\n"
        "- Display math uses $$...$$.\n"
        "- Remove malformed wrappers like [ equation ] or (equation).\n"
        "- Keep all source citations exactly as provided.\n\n"
        "Output rules:\n"
        "- Return only the corrected answer body.\n"
        "- Do not include prefaces like 'To repair...' or 'Final answer:'.\n\n"
        "Answer to repair:\n{answer}\n\n"
        "Return only the repaired answer."
    )
    repair_chain = repair_prompt | LLM
    repaired = repair_chain.invoke({"answer": answer})
    text = str(repaired).strip()
    if text.startswith("Answer to repair:"):
        text = text.split("Answer to repair:", 1)[1].lstrip()
    return text


def _normalize_math_delimiters(text: str) -> str:
    # Convert equation-like bracket wrappers to display math.
    text = re.sub(
        r"\[\s*([^\]\n]*(?:=|\\frac|\\Theta|\\Omega|\\log|[\^_])[^\]\n]*)\s*\]",
        lambda m: f"$$\n{m.group(1).strip()}\n$$",
        text,
    )

    # Do not rewrite parentheses-delimited LaTeX (e.g., \left( ... \right)):
    # this was corrupting valid math expressions.
    return text


def _strip_repair_preface(text: str) -> str:
    text = re.sub(
        r"^\s*To repair.*?(?:final answer is:|The final answer is:)\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"^\s*Final answer:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_latex_escapes(text: str) -> str:
    # Convert escaped LaTeX delimiters to active delimiters.
    text = text.replace(r"\$$", "$$")
    text = text.replace(r"\$", "$")

    # Convert \( ... \) and \[ ... \] delimiters to the ones configured in Gradio.
    text = re.sub(
        r"\\\((.+?)\\\)",
        lambda m: f"${m.group(1).strip()}$",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\[(.+?)\\\]",
        lambda m: f"$$\n{m.group(1).strip()}\n$$",
        text,
        flags=re.DOTALL,
    )
    return text


def _strip_generated_sources_section(text: str) -> str:
    # Remove trailing model-generated source lists; app renders canonical sources separately.
    text = re.sub(
        r"\n{0,2}Sources:\s*(?:\n\s*[-*]?\s*\[\d+\][^\n]*)+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\n{0,2}Sources:\s*(?:\n\s*\[\d+\]\s*)+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.rstrip()


def answer_query(
    query: str, retriever, chat_history: list | None = None
) -> tuple[str, dict[str, str]]:
    docs = retriever.invoke(query)
    context, sources = _format_docs(docs)
    history = _format_history(chat_history)

    prompt = PromptTemplate.from_template(
        "{system_prompt}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Respond in Markdown only.\n"
        "Math rules:\n"
        "- Inline math must use $...$.\n"
        "- Display math must use $$...$$ on separate lines.\n"
        "- Do not wrap equations in square brackets like [ ... ].\n"
        "- Do not add redundant parentheses around standalone symbols.\n"
        "- Use inline citations like [1], [2] where relevant.\n"
        "- Do not output a separate 'Sources:' section."
    )

    chain = prompt | LLM
    response = chain.invoke(
        {
            "system_prompt": config.SYSTEM_PROMPT,
            "history": history,
            "question": query,
            "context": context,
        }
    )
    answer = str(response).strip()

    # Second pass: if math is present, run formatter repair. Retry once if still malformed.
    if _contains_math(answer):
        for _ in range(2):
            answer = _repair_math_format(answer)
            answer = _normalize_math_delimiters(answer)
            answer = _normalize_latex_escapes(answer)
            if not _has_math_format_issues(answer):
                break

    answer = _strip_repair_preface(answer)
    answer = _normalize_latex_escapes(answer)
    answer = _strip_generated_sources_section(answer)

    return answer, sources
