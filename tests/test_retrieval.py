from unittest.mock import patch

from rag_eval_gate.retrieval import Index, similarity

CHUNKS = [
    "Refunds\nMoney back within 14 days.",
    "Login problems\nUpdate the app.",
    "Office\nWe are in Abu Dhabi.",
]

# Toy embeddings: [money, tech, place]
TOY = {
    CHUNKS[0]: [0.9, 0.1, 0.0],
    CHUNKS[1]: [0.1, 0.9, 0.0],
    CHUNKS[2]: [0.0, 0.1, 0.9],
    "money back?": [0.9, 0.0, 0.1],
    "weather?": [0.1, 0.1, 0.3],
}


def fake_embed(texts):
    return [TOY[t] for t in texts]


def test_similarity_is_dot_product():
    assert similarity([1, 2], [3, 4]) == 11  # 1*3 + 2*4


def test_best_chunk_comes_first():
    with patch("rag_eval_gate.retrieval.embed", side_effect=fake_embed):
        hits = Index(CHUNKS).search("money back?", k=2, min_score=0)
    assert hits[0].title == "Refunds"
    assert len(hits) == 2


def test_k_limits_number_of_results():
    with patch("rag_eval_gate.retrieval.embed", side_effect=fake_embed):
        hits = Index(CHUNKS).search("money back?", k=1, min_score=0)
    assert len(hits) == 1


def test_weak_matches_are_dropped():
    with patch("rag_eval_gate.retrieval.embed", side_effect=fake_embed):
        hits = Index(CHUNKS).search("weather?", k=2, min_score=0.4)
    assert hits == []
