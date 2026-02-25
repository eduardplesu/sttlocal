"""
Transcript analysis using:
- Azure OpenAI (GPT-4o deployment) via the OpenAI Python SDK (new or legacy style)
- Phi-4 (or any chat model endpoint) via azure.ai.inference ChatCompletionsClient

Returns a Romanian, structured analysis:
1) Comprehensive Summary
2) Key Discussion Points
3) Decisions Made
4) Action Items (owner + deadline if available)
5) Follow-up Questions / Risks
6) Sentiment and Collaboration Tone

Never invent facts if missing from transcript.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import openai
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()


Engine = Literal["gpt4o", "phi4"]

DEFAULT_AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
DEFAULT_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

SYSTEM_PROMPT_RO = (
    "You are an assistant that analyzes meeting or conversation transcripts from any domain. "
    "Create a structured, practical output in Romanian with these sections: "
    "1) Comprehensive Summary, "
    "2) Key Discussion Points, "
    "3) Decisions Made, "
    "4) Action Items (owner + deadline if available), "
    "5) Follow-up Questions / Risks, "
    "6) Sentiment and Collaboration Tone. "
    "If details are missing, say so explicitly and do not invent facts."
)


def _require_env(name: str, value: Optional[str]) -> str:
    if value is None or not value.strip():
        raise ValueError(f"Missing environment variable: {name}")
    return value.strip()


def _normalize_endpoint(endpoint: str) -> str:
    # Keep it stable (no trailing slash).
    return endpoint.strip().rstrip("/")


def _normalize_phi_endpoint(endpoint: str) -> str:
    """
    Accepts endpoints that may include common suffixes and normalizes to the base.
    Examples seen in the wild:
      - https://.../v1/chat/completions
      - https://.../chat/completions
    """
    ep = _normalize_endpoint(endpoint)
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if ep.endswith(suffix):
            ep = ep[: -len(suffix)]
            ep = _normalize_endpoint(ep)
    return ep


def _build_messages(transcription_text: str) -> list[dict]:
    text = (transcription_text or "").strip()
    if not text:
        raise ValueError("transcription_text is empty.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT_RO},
        {"role": "user", "content": f"Analyze the following transcription:\n\n{text}"},
    ]


def _analyze_with_azure_openai(messages: list[dict]) -> str:
    """
    Uses the OpenAI Python SDK in a version-tolerant way:
    - openai>=1.x: AzureOpenAI client
    - openai<1.x (legacy): openai.ChatCompletion.create + global config
    """
    endpoint = _require_env("OPENAI_ENDPOINT", os.getenv("OPENAI_ENDPOINT"))
    api_key = _require_env("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    deployment = os.getenv("DEPLOYMENT_NAME", DEFAULT_DEPLOYMENT_NAME).strip()
    api_version = os.getenv("OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION).strip()

    endpoint = _normalize_endpoint(endpoint)

    # Try modern SDK first (openai>=1.x)
    try:
        from openai import AzureOpenAI  # type: ignore

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )

        content = (resp.choices[0].message.content or "").strip()
        return content

    except Exception:
        # Fall back to legacy style (openai<1.x)
        openai.api_type = "azure"
        openai.api_base = endpoint
        openai.api_version = api_version
        openai.api_key = api_key

        resp = openai.ChatCompletion.create(
            engine=deployment,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )

        content = (resp.choices[0].message["content"] or "").strip()
        return content


def _analyze_with_phi4(messages: list[dict]) -> str:
    phi_endpoint = _require_env("PHI4_ENDPOINT", os.getenv("PHI4_ENDPOINT"))
    phi_key = _require_env("PHI4_KEY", os.getenv("PHI4_KEY"))

    phi_endpoint = _normalize_phi_endpoint(phi_endpoint)

    client = ChatCompletionsClient(
        endpoint=phi_endpoint,
        credential=AzureKeyCredential(phi_key),
    )

    payload = {
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    resp = client.complete(payload)
    content = (resp.choices[0].message.content or "").strip()
    return content


def analyze_transcription(transcription_text: str, engine: Engine = "gpt4o") -> str:
    """
    Analyze a transcript using either:
      - engine="gpt4o": Azure OpenAI deployment
      - engine="phi4": Azure AI Inference chat endpoint (Phi-4)

    Returns Romanian, structured analysis text.
    """
    messages = _build_messages(transcription_text)

    if engine == "gpt4o":
        return _analyze_with_azure_openai(messages)

    if engine == "phi4":
        return _analyze_with_phi4(messages)

    raise ValueError("Unsupported engine. Choose 'gpt4o' or 'phi4'.")