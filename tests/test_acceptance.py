"""Acceptance tests: verify end-to-end clustering behavior with 10+ ideas across both engines."""

import numpy as np
from unittest.mock import MagicMock, patch

from tests.conftest import (
    add_idea,
    make_mock_embedding_response,
    mock_openai_chat_response,
    mock_openai_response,
)


# A set of 12 ideas spanning 4 topic groups:
# Group A: User onboarding (3 ideas)
# Group B: Dark mode / UI themes (3 ideas)
# Group C: Email notifications (3 ideas)
# Group D: Search functionality (3 ideas)
IDEAS = [
    # Group A - User onboarding
    ("Improve the new user onboarding flow", "Onboarding"),
    ("Make the signup experience smoother for first-time users", "Onboarding"),
    ("Add a guided walkthrough for new users after registration", "Onboarding"),
    # Group B - Dark mode / UI themes
    ("Add a dark mode toggle to the settings page", "Dark Mode"),
    ("Support dark and light theme switching in the UI", "Dark Mode"),
    ("Let users choose their preferred color theme", "Dark Mode"),
    # Group C - Email notifications
    ("Send email notifications when a user account is updated", "Email Notifications"),
    ("Notify users by email when their password is changed", "Email Notifications"),
    ("Add configurable email alerts for account changes", "Email Notifications"),
    # Group D - Search functionality
    ("Add a search bar to the main navigation header", "Search"),
    ("Implement full-text search across all stored ideas", "Search"),
    ("Build an autocomplete search feature for idea lookup", "Search"),
]

# Embedding vectors per topic group (similar within group, distant between groups)
TOPIC_EMBEDDINGS = {
    "onboarding": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
    "signup": np.array([0.88, 0.12, 0.01, 0.0], dtype=np.float32),
    "walkthrough": np.array([0.87, 0.13, 0.02, 0.0], dtype=np.float32),
    "dark mode": np.array([0.0, 0.9, 0.0, 0.0], dtype=np.float32),
    "theme switching": np.array([0.01, 0.88, 0.02, 0.0], dtype=np.float32),
    "color theme": np.array([0.02, 0.87, 0.01, 0.0], dtype=np.float32),
    "email notifications": np.array([0.0, 0.0, 0.9, 0.0], dtype=np.float32),
    "email when": np.array([0.01, 0.0, 0.88, 0.02], dtype=np.float32),
    "email alerts": np.array([0.02, 0.01, 0.87, 0.0], dtype=np.float32),
    "search bar": np.array([0.0, 0.0, 0.0, 0.9], dtype=np.float32),
    "full-text search": np.array([0.0, 0.02, 0.01, 0.88], dtype=np.float32),
    "autocomplete search": np.array([0.01, 0.0, 0.02, 0.87], dtype=np.float32),
}


def _make_bertopic_mock():
    """Create a mock OpenAI client for bertopic with topic-aware embeddings."""
    mock_client = MagicMock()

    def fake_embeddings_create(**kwargs):
        text = kwargs.get("input", [""])[0].lower()
        for substring, emb in TOPIC_EMBEDDINGS.items():
            if substring in text:
                return make_mock_embedding_response(emb)
        return make_mock_embedding_response(np.random.rand(4).astype(np.float32))

    mock_client.embeddings.create.side_effect = fake_embeddings_create
    mock_client.chat.completions.create.return_value = mock_openai_chat_response("Topic")
    return mock_client


@patch("cluster_api.engines.bertopic_engine._get_client")
@patch("cluster_api.engines.llm_engine._get_client")
def test_cluster_10_plus_ideas_with_both_engines(mock_llm_client, mock_bert_client, client):
    """POST 10+ ideas, cluster each with both bertopic and llm, verify clusters form sensibly."""
    mock_bert_client.return_value = _make_bertopic_mock()

    mock_llm = MagicMock()
    mock_llm_client.return_value = mock_llm

    idea_ids = []
    for text, _ in IDEAS:
        idea_ids.append(add_idea(client, text))

    assert len(idea_ids) == 12

    # Cluster each idea with BERTopic
    bertopic_results = []
    for idea_id in idea_ids:
        resp = client.post("/cluster/bertopic", json={"idea_id": idea_id})
        assert resp.status_code == 200
        bertopic_results.append(resp.json())

    # Cluster each idea with LLM (mock responses based on expected group)
    llm_results = []
    for i, (text, group_name) in enumerate(IDEAS):
        is_first_in_group = i % 3 == 0
        mock_llm.chat.completions.create.return_value = mock_openai_response(
            group_name, is_new=is_first_in_group
        )
        resp = client.post("/cluster/llm", json={"idea_id": idea_ids[i]})
        assert resp.status_code == 200
        llm_results.append(resp.json())

    # Verify BERTopic clusters exist and ideas are assigned
    resp = client.get("/clusters/bertopic")
    assert resp.status_code == 200
    bertopic_clusters = resp.json()
    assert len(bertopic_clusters) >= 1
    bertopic_total = sum(len(c["ideas"]) for c in bertopic_clusters)
    assert bertopic_total == 12

    # Verify LLM clusters: should have exactly 4 clusters (one per group)
    resp = client.get("/clusters/llm")
    assert resp.status_code == 200
    llm_clusters = resp.json()
    assert len(llm_clusters) == 4
    llm_total = sum(len(c["ideas"]) for c in llm_clusters)
    assert llm_total == 12

    # Each LLM cluster should have exactly 3 ideas
    for cluster in llm_clusters:
        assert len(cluster["ideas"]) == 3

    # Verify cluster names match expected groups
    llm_names = {c["name"] for c in llm_clusters}
    assert llm_names == {"Onboarding", "Dark Mode", "Email Notifications", "Search"}


@patch("cluster_api.engines.bertopic_engine._get_client")
def test_new_idea_joins_existing_cluster_bertopic(mock_get_client, client):
    """Add an idea that should fit an existing BERTopic cluster rather than creating a new one."""
    onboarding_emb = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    onboarding_emb_2 = np.array([0.88, 0.12, 0.01], dtype=np.float32)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_chat_response("Onboarding")

    def fake_embeddings_create(**kwargs):
        text = kwargs.get("input", [""])[0].lower()
        if "better" in text:
            return make_mock_embedding_response(onboarding_emb_2)
        return make_mock_embedding_response(onboarding_emb)

    mock_client.embeddings.create.side_effect = fake_embeddings_create
    mock_get_client.return_value = mock_client

    # Create a cluster with a seed idea
    id1 = add_idea(client, "Improve the user onboarding experience for new signups")
    r1 = client.post("/cluster/bertopic", json={"idea_id": id1})
    assert r1.status_code == 200
    assert r1.json()["is_new"] is True
    original_cluster_id = r1.json()["cluster_id"]

    # Add a very similar idea - should join the existing cluster
    id2 = add_idea(client, "Make the onboarding experience better for new signups")
    r2 = client.post("/cluster/bertopic", json={"idea_id": id2})
    assert r2.status_code == 200
    assert r2.json()["is_new"] is False
    assert r2.json()["cluster_id"] == original_cluster_id


@patch("cluster_api.engines.llm_engine._get_client")
def test_new_idea_joins_existing_cluster_llm(mock_get_client, client):
    """Add an idea that should fit an existing LLM cluster rather than creating a new one."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Create a cluster with a seed idea
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "User Onboarding", is_new=True
    )
    id1 = add_idea(client, "Improve the user onboarding experience")
    r1 = client.post("/cluster/llm", json={"idea_id": id1})
    assert r1.status_code == 200
    assert r1.json()["is_new"] is True
    original_cluster_id = r1.json()["cluster_id"]

    # Add a new idea that the LLM assigns to the existing cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "User Onboarding", is_new=False
    )
    id2 = add_idea(client, "Add a guided tour for first-time users")
    r2 = client.post("/cluster/llm", json={"idea_id": id2})
    assert r2.status_code == 200
    assert r2.json()["is_new"] is False
    assert r2.json()["cluster_id"] == original_cluster_id

    # Verify the cluster now has 2 ideas
    resp = client.get("/clusters/llm")
    assert resp.status_code == 200
    clusters = resp.json()
    assert len(clusters) == 1
    assert len(clusters[0]["ideas"]) == 2
