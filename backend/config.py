# backend/config.py — LLM and embedding provider switcher (Phase 0)
# Switching providers: set LLM_PROVIDER / EMBEDDING_PROVIDER in .env only.
import os


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
