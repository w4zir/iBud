"""Runtime configuration and shared logging helpers."""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict


def is_debug() -> bool:
    """Return True if DEBUG env is set to a truthy value (true, 1, yes). Default: False."""
    raw = (os.getenv("DEBUG") or "").strip().lower()
    return raw in ("true", "1", "yes")


class _StructuredFormatter(logging.Formatter):
    REDACT_KEYS = {"password", "token", "api_key", "secret", "authorization"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra_fields = getattr(record, "structured_extra", {})
        if isinstance(extra_fields, dict):
            payload.update(_redact_dict(extra_fields))
        return json.dumps(payload, ensure_ascii=True)


def _redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    redacted: Dict[str, Any] = {}
    for key, value in data.items():
        key_l = str(key).lower()
        if key_l in _StructuredFormatter.REDACT_KEYS:
            redacted[key] = "***REDACTED***"
            continue
        if isinstance(value, dict):
            redacted[key] = _redact_dict(value)
        else:
            redacted[key] = value
    return redacted


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(_StructuredFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG if is_debug() else logging.INFO)
    return logger


def log_event(
    component: str,
    message: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    logger = get_logger(component)
    logger.log(level, message, extra={"structured_extra": fields})


def debug_print(tag: str, message: str, **kwargs: object) -> None:
    """Backward-compatible debug logger wrapper."""
    if not is_debug():
        return
    log_event(tag, message, level=logging.DEBUG, **kwargs)


def get_embedding_dim() -> int:
    """Return embedding dimension; must match VECTOR(n) in schema. Default 768 (Ollama)."""
    return int(os.getenv("EMBEDDING_DIM", "768"))


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    match provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        case "cerebras":
            try:
                from langchain_cerebras import ChatCerebras
            except ImportError:
                raise ImportError(
                    "LLM_PROVIDER=cerebras requires langchain-cerebras. "
                    "Install with: pip install langchain-cerebras"
                )
            return ChatCerebras(
                model=os.getenv("CEREBRAS_MODEL", "llama3.1-8b"),
                api_key=os.getenv("CEREBRAS_API_KEY"),
            )
        case "ollama" | _:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )


def get_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-3-small")
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_OLLAMA", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
