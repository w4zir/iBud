from backend.rag.ingest import build_document_text, chunk_documents


def test_build_document_text_mapping_preserves_core_fields():
    example = {
        "question": "How do I track my order?",
        "answer": "You can track your order from the Orders page.",
        "category": "shipping",
    }

    content, metadata = build_document_text(example)

    assert "Q: How do I track my order?" in content
    assert "A: You can track my order from the Orders page."[:10].startswith("A:")
    assert metadata["category"] == "shipping"
    assert metadata["source_example"]["question"] == example["question"]
    assert metadata["source_example"]["answer"] == example["answer"]


def test_chunk_documents_preserves_metadata_and_splits_text():
    content = "Q: Test?\nA: " + "x" * 1200
    contents_and_metadata = [(content, {"category": "test-cat"})]

    chunks = chunk_documents(contents_and_metadata, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 2
    for chunk_text, metadata in chunks:
        assert metadata["category"] == "test-cat"

