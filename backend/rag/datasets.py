from __future__ import annotations

"""
Lightweight registry for knowledge-base datasets used by the RAG pipeline.

This centralises dataset keys so that ingestion, retrieval, and evaluation
can agree on stable identifiers without hard-coding them in multiple places.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    source: str
    description: str


DATASETS: Dict[str, DatasetConfig] = {
    # Primary KB corpus: Wix/WixQA articles ingested via ingest_wixqa.
    "wixqa": DatasetConfig(
        key="wixqa",
        source="wixqa",
        description="Wix/WixQA knowledge base articles",
    ),
    # Secondary corpus: Bitext customer-support QA pairs ingested via ingest_bitext.
    "bitext": DatasetConfig(
        key="bitext",
        source="bitext",
        description="Bitext customer-support QA pairs",
    ),
    # Foodpanda policy markdown corpus ingested via ingest_foodpanda.
    "foodpanda": DatasetConfig(
        key="foodpanda",
        source="foodpanda",
        description="Foodpanda e-commerce policy documentation",
    ),
}


def get_dataset_config(key: str) -> DatasetConfig:
    """
    Return the DatasetConfig for a given key, raising KeyError if unknown.
    """
    return DATASETS[key]


__all__ = ["DatasetConfig", "DATASETS", "get_dataset_config"]

