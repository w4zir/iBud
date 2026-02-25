import os
from typing import Iterable, List

from ..config import get_embedding_model, get_embedding_dim


class EmbeddingClient:
    """
    Thin wrapper around the LangChain embedding model returned by get_embedding_model().

    Exposes a minimal, provider-agnostic interface used by the ingest pipeline.
    Embedding dimension is read from EMBEDDING_DIM env (default 768).
    """

    def __init__(self) -> None:
        self._model = get_embedding_model()

    @property
    def provider(self) -> str:
        return os.getenv("EMBEDDING_PROVIDER", "ollama")

    @property
    def dimension(self) -> int:
        return get_embedding_dim()

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self._model.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)


def get_client() -> EmbeddingClient:
    """
    Factory for obtaining an EmbeddingClient.
    """
    return EmbeddingClient()


__all__ = ["EmbeddingClient", "get_client"]

