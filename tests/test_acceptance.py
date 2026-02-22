"""Acceptance tests: verify end-to-end clustering behavior with 10+ ideas across both engines."""

from unittest.mock import MagicMock, patch

import cluster_api.engines.bertopic_engine as bertopic_engine
from tests.conftest import add_idea, mock_openai_response


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


@patch("cluster_api.engines.llm_engine._get_client")
def test_cluster_10_plus_ideas_with_both_engines(mock_get_client, client):
    """POST 10+ ideas, cluster each with both bertopic and llm, verify clusters form sensibly."""
    bertopic_engine._model = None

    # Setup LLM mock to return topic-appropriate clusters
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

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
        # For each group, the first idea creates a new cluster, the rest join it
        is_first_in_group = i % 3 == 0
        mock_client.chat.completions.create.return_value = mock_openai_response(
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


def test_new_idea_joins_existing_cluster_bertopic(client):
    """Add an idea that should fit an existing BERTopic cluster rather than creating a new one."""
    bertopic_engine._model = None

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
