"""Integration tests for BERTopic clustering endpoints (OpenAI embeddings)."""

import numpy as np
from unittest.mock import MagicMock, patch

from tests.conftest import add_idea, make_mock_embedding_response, mock_openai_chat_response


# Deterministic embeddings for testing: similar ideas get similar vectors
ONBOARDING_EMB = np.array([0.9, 0.1, 0.0], dtype=np.float32)
ONBOARDING_EMB_2 = np.array([0.88, 0.12, 0.01], dtype=np.float32)
DARK_MODE_EMB = np.array([0.0, 0.1, 0.9], dtype=np.float32)
ALERTS_EMB = np.array([0.1, 0.9, 0.0], dtype=np.float32)

_call_count = 0


def _make_mock_client(embedding_map=None, default_emb=None, cluster_name="Topic"):
    """Create a mock OpenAI client with embeddings and chat completions.

    Args:
        embedding_map: dict mapping idea text substrings to embeddings.
        default_emb: fallback embedding if no match found.
        cluster_name: name returned by cluster name generation.
    """
    mock_client = MagicMock()

    def fake_embeddings_create(**kwargs):
        text = kwargs.get("input", [""])[0]
        if embedding_map:
            for substring, emb in embedding_map.items():
                if substring.lower() in text.lower():
                    return make_mock_embedding_response(emb)
        emb = default_emb if default_emb is not None else np.random.rand(3).astype(np.float32)
        return make_mock_embedding_response(emb)

    mock_client.embeddings.create.side_effect = fake_embeddings_create
    mock_client.chat.completions.create.return_value = mock_openai_chat_response(cluster_name)

    return mock_client


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_cluster_single_idea(mock_get_client, client):
    mock_get_client.return_value = _make_mock_client(
        default_emb=ONBOARDING_EMB, cluster_name="User Onboarding"
    )
    idea_id = add_idea(client, "We should improve our onboarding flow for new users")
    resp = client.post("/cluster/bertopic", json={"idea_id": idea_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["idea_id"] == idea_id
    assert "cluster_id" in data
    assert data["cluster_name"] == "User Onboarding"
    assert data["is_new"] is True
    assert data["confidence"] == 1.0


def test_cluster_nonexistent_idea(client):
    resp = client.post("/cluster/bertopic", json={"idea_id": 9999})
    assert resp.status_code == 404


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_similar_ideas_same_cluster(mock_get_client, client):
    mock_client = _make_mock_client(
        embedding_map={
            "onboarding experience": ONBOARDING_EMB,
            "onboarding flow": ONBOARDING_EMB_2,
        },
        cluster_name="User Onboarding",
    )
    mock_get_client.return_value = mock_client

    id1 = add_idea(client, "Improve the user onboarding experience")
    id2 = add_idea(client, "Make the onboarding flow better for new users")

    r1 = client.post("/cluster/bertopic", json={"idea_id": id1})
    assert r1.status_code == 200
    cluster1 = r1.json()

    r2 = client.post("/cluster/bertopic", json={"idea_id": id2})
    assert r2.status_code == 200
    cluster2 = r2.json()

    assert cluster1["cluster_id"] == cluster2["cluster_id"]
    assert cluster2["is_new"] is False
    assert cluster2["confidence"] > 0.0


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_different_ideas_different_clusters(mock_get_client, client):
    mock_client = _make_mock_client(
        embedding_map={
            "dark mode": DARK_MODE_EMB,
            "stock price": ALERTS_EMB,
        },
        cluster_name="Topic",
    )
    mock_get_client.return_value = mock_client

    id1 = add_idea(client, "We need a dark mode toggle in the settings page")
    id2 = add_idea(client, "Add real-time stock price alerts via SMS notifications")

    r1 = client.post("/cluster/bertopic", json={"idea_id": id1})
    assert r1.status_code == 200
    cluster1 = r1.json()

    r2 = client.post("/cluster/bertopic", json={"idea_id": id2})
    assert r2.status_code == 200
    cluster2 = r2.json()

    assert cluster1["cluster_id"] != cluster2["cluster_id"]
    assert cluster1["is_new"] is True
    assert cluster2["is_new"] is True


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_get_bertopic_clusters(mock_get_client, client):
    mock_client = _make_mock_client(
        embedding_map={
            "search bar": ONBOARDING_EMB,
            "search feature": ONBOARDING_EMB_2,
            "recommendation": DARK_MODE_EMB,
        },
        cluster_name="Search Feature",
    )
    mock_get_client.return_value = mock_client

    id1 = add_idea(client, "Add a search bar to the navigation header")
    id2 = add_idea(client, "Include a search feature in the top navigation")
    id3 = add_idea(client, "Build a recommendation engine for product suggestions")

    client.post("/cluster/bertopic", json={"idea_id": id1})
    client.post("/cluster/bertopic", json={"idea_id": id2})
    client.post("/cluster/bertopic", json={"idea_id": id3})

    resp = client.get("/clusters/bertopic")
    assert resp.status_code == 200
    clusters = resp.json()
    assert len(clusters) >= 1

    for cluster in clusters:
        assert "cluster_id" in cluster
        assert "name" in cluster
        assert "ideas" in cluster
        assert len(cluster["ideas"]) >= 1
        for idea in cluster["ideas"]:
            assert "id" in idea
            assert "text" in idea
            assert "user_id" in idea

    total_ideas = sum(len(c["ideas"]) for c in clusters)
    assert total_ideas == 3


def test_get_bertopic_clusters_empty(client):
    resp = client.get("/clusters/bertopic")
    assert resp.status_code == 200
    assert resp.json() == []


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_cluster_multiple_ideas_incrementally(mock_get_client, client):
    emb_base = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    emb_sim1 = np.array([0.89, 0.11, 0.01], dtype=np.float32)
    emb_sim2 = np.array([0.88, 0.12, 0.02], dtype=np.float32)

    mock_client = _make_mock_client(
        embedding_map={
            "modified": emb_base,
            "updated": emb_sim1,
            "changed": emb_sim2,
        },
        cluster_name="Email Notifications",
    )
    mock_get_client.return_value = mock_client

    ideas = [
        "Add email notifications when user accounts are modified",
        "Send email notifications when user accounts are updated",
        "Notify users via email when their account is changed",
    ]
    idea_ids = [add_idea(client, text) for text in ideas]

    results = []
    for idea_id in idea_ids:
        r = client.post("/cluster/bertopic", json={"idea_id": idea_id})
        assert r.status_code == 200
        results.append(r.json())

    assert results[0]["is_new"] is True
    joined_existing = any(not r["is_new"] for r in results[1:])
    assert joined_existing, "Expected at least one similar idea to join an existing cluster"


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_cluster_duplicate_idea_rejected(mock_get_client, client):
    mock_get_client.return_value = _make_mock_client(
        default_emb=ONBOARDING_EMB, cluster_name="Onboarding"
    )
    idea_id = add_idea(client, "Improve the user onboarding experience")
    resp = client.post("/cluster/bertopic", json={"idea_id": idea_id})
    assert resp.status_code == 200

    resp = client.post("/cluster/bertopic", json={"idea_id": idea_id})
    assert resp.status_code == 409
