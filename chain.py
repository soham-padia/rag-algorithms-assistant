"""Prompt and LLM chain utilities for grounded QA."""

from __future__ import annotations

from collections import OrderedDict
import os
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


def answer_query(query: str, retriever) -> tuple[str, dict[str, str]]:
    docs = retriever.invoke(query)
    context, sources = _format_docs(docs)

    prompt = PromptTemplate.from_template(
        "{system_prompt}\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Respond with a concise explanation and include citation ids."
    )

    chain = prompt | LLM
    response = chain.invoke(
        {"system_prompt": config.SYSTEM_PROMPT, "question": query, "context": context}
    )
    return str(response).strip(), sources
