"""Integration tests for BERTopic clustering endpoints."""

import cluster_api.engines.bertopic_engine as bertopic_engine


def _add_idea(client, text, user_id="u1"):
    resp = client.post("/ideas", json={"text": text, "user_id": user_id})
    assert resp.status_code == 201
    return resp.json()["idea_id"]


def _reset_model():
    """Reset the cached sentence-transformer model between tests."""
    bertopic_engine._model = None


def test_cluster_single_idea(client):
    _reset_model()
    idea_id = _add_idea(client, "We should improve our onboarding flow for new users")
    resp = client.post("/cluster/bertopic", json={"idea_id": idea_id})
    assert resp.status_code == 200
    data = resp.json()
    assert "cluster_id" in data
    assert "cluster_name" in data
    assert data["is_new"] is True


def test_cluster_nonexistent_idea(client):
    resp = client.post("/cluster/bertopic", json={"idea_id": 9999})
    assert resp.status_code == 404


def test_similar_ideas_same_cluster(client):
    _reset_model()
    id1 = _add_idea(client, "Improve the user onboarding experience")
    id2 = _add_idea(client, "Make the onboarding flow better for new users")

    r1 = client.post("/cluster/bertopic", json={"idea_id": id1})
    assert r1.status_code == 200
    cluster1 = r1.json()

    r2 = client.post("/cluster/bertopic", json={"idea_id": id2})
    assert r2.status_code == 200
    cluster2 = r2.json()

    # These very similar ideas should end up in the same cluster
    assert cluster1["cluster_id"] == cluster2["cluster_id"]
    assert cluster2["is_new"] is False


def test_different_ideas_different_clusters(client):
    _reset_model()
    id1 = _add_idea(client, "We need a dark mode toggle in the settings page")
    id2 = _add_idea(client, "Add real-time stock price alerts via SMS notifications")

    r1 = client.post("/cluster/bertopic", json={"idea_id": id1})
    assert r1.status_code == 200
    cluster1 = r1.json()

    r2 = client.post("/cluster/bertopic", json={"idea_id": id2})
    assert r2.status_code == 200
    cluster2 = r2.json()

    # These dissimilar ideas should be in different clusters
    assert cluster1["cluster_id"] != cluster2["cluster_id"]
    assert cluster1["is_new"] is True
    assert cluster2["is_new"] is True


def test_get_bertopic_clusters(client):
    _reset_model()
    id1 = _add_idea(client, "Add a search bar to the navigation header")
    id2 = _add_idea(client, "Include a search feature in the top navigation")
    id3 = _add_idea(client, "Build a recommendation engine for product suggestions")

    client.post("/cluster/bertopic", json={"idea_id": id1})
    client.post("/cluster/bertopic", json={"idea_id": id2})
    client.post("/cluster/bertopic", json={"idea_id": id3})

    resp = client.get("/clusters/bertopic")
    assert resp.status_code == 200
    clusters = resp.json()
    assert len(clusters) >= 1  # At least one cluster exists

    # Verify cluster structure
    for cluster in clusters:
        assert "cluster_id" in cluster
        assert "name" in cluster
        assert "ideas" in cluster
        assert len(cluster["ideas"]) >= 1
        for idea in cluster["ideas"]:
            assert "id" in idea
            assert "text" in idea
            assert "user_id" in idea

    # Total ideas across all clusters should be 3
    total_ideas = sum(len(c["ideas"]) for c in clusters)
    assert total_ideas == 3


def test_get_bertopic_clusters_empty(client):
    resp = client.get("/clusters/bertopic")
    assert resp.status_code == 200
    assert resp.json() == []


def test_cluster_multiple_ideas_incrementally(client):
    _reset_model()
    # Add several ideas on the same topic incrementally (high similarity phrases)
    ideas = [
        "Add email notifications when user accounts are modified",
        "Send email notifications when user accounts are updated",
        "Notify users via email when their account is changed",
    ]

    idea_ids = [_add_idea(client, text) for text in ideas]

    # Cluster them one by one
    results = []
    for idea_id in idea_ids:
        r = client.post("/cluster/bertopic", json={"idea_id": idea_id})
        assert r.status_code == 200
        results.append(r.json())

    # First one creates a new cluster
    assert results[0]["is_new"] is True

    # At least one of the subsequent ones should join the existing cluster
    joined_existing = any(not r["is_new"] for r in results[1:])
    assert joined_existing, "Expected at least one similar idea to join an existing cluster"
