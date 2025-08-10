import pytest

def test_import_embeddings_module():
    try:
        import core.stage1_qa.qa_embed as qa_embed
    except Exception as e:
        pytest.fail(f"Failed to import qa_embed: {e}")

def test_topk_embeddings_when_available():
    # skip if embeddings libs not installed
    try:
        import sentence_transformers, faiss
    except Exception:
        pytest.skip("Embeddings libraries not installed; skipping embeddings test.")

    from core.stage1_qa.qa_embed import topk_passages_embed
    docs = [
        {"id":"d1.txt","company":"X","period":"2023","text":"Risks include currency fluctuations and supply issues.","location_hint":"file:d1.txt"},
        {"id":"d2.txt","company":"X","period":"2024","text":"We achieved strong growth; outlook remains positive.","location_hint":"file:d2.txt"}
    ]
    topk = topk_passages_embed("What risks were mentioned?", docs, k=2)
    assert isinstance(topk, list)
    assert len(topk) >= 1
