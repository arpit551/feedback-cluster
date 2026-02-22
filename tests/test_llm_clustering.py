"""Integration tests for LLM clustering endpoints."""

from unittest.mock import MagicMock, patch

from tests.conftest import add_idea, mock_openai_response


@patch("cluster_api.engines.llm_engine._get_client")
def test_cluster_single_idea_new(mock_get_client, client):
    """A single idea with no existing clusters should create a new cluster."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "User Onboarding", is_new=True
    )
    mock_get_client.return_value = mock_client

    idea_id = add_idea(client, "We should improve our onboarding flow for new users")
    resp = client.post("/cluster/llm", json={"idea_id": idea_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cluster_name"] == "User Onboarding"
    assert data["is_new"] is True
    assert data["idea_id"] == idea_id
    assert "cluster_id" in data

    # Verify the OpenAI API was called
    mock_client.chat.completions.create.assert_called_once()


@patch("cluster_api.engines.llm_engine._get_client")
def test_cluster_nonexistent_idea(mock_get_client, client):
    """Clustering a non-existent idea should return 404."""
    resp = client.post("/cluster/llm", json={"idea_id": 9999})
    assert resp.status_code == 404


@patch("cluster_api.engines.llm_engine._get_client")
def test_similar_ideas_same_cluster(mock_get_client, client):
    """Two similar ideas should end up in the same cluster when LLM assigns them together."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # First idea: LLM creates a new cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "User Onboarding", is_new=True
    )
    id1 = add_idea(client, "Improve the user onboarding experience")
    r1 = client.post("/cluster/llm", json={"idea_id": id1})
    assert r1.status_code == 200
    cluster1 = r1.json()
    assert cluster1["is_new"] is True

    # Second idea: LLM assigns to existing cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "User Onboarding", is_new=False
    )
    id2 = add_idea(client, "Make the onboarding flow better for new users")
    r2 = client.post("/cluster/llm", json={"idea_id": id2})
    assert r2.status_code == 200
    cluster2 = r2.json()

    assert cluster1["cluster_id"] == cluster2["cluster_id"]
    assert cluster2["is_new"] is False


@patch("cluster_api.engines.llm_engine._get_client")
def test_different_ideas_different_clusters(mock_get_client, client):
    """Two dissimilar ideas should end up in different clusters."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # First idea: new cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Dark Mode", is_new=True
    )
    id1 = add_idea(client, "We need a dark mode toggle in the settings page")
    r1 = client.post("/cluster/llm", json={"idea_id": id1})
    assert r1.status_code == 200
    cluster1 = r1.json()

    # Second idea: different new cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Stock Alerts", is_new=True
    )
    id2 = add_idea(client, "Add real-time stock price alerts via SMS notifications")
    r2 = client.post("/cluster/llm", json={"idea_id": id2})
    assert r2.status_code == 200
    cluster2 = r2.json()

    assert cluster1["cluster_id"] != cluster2["cluster_id"]
    assert cluster1["is_new"] is True
    assert cluster2["is_new"] is True


@patch("cluster_api.engines.llm_engine._get_client")
def test_get_llm_clusters(mock_get_client, client):
    """GET /clusters/llm should return all LLM clusters with their ideas."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Create two ideas in the same cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Search Feature", is_new=True
    )
    id1 = add_idea(client, "Add a search bar to the navigation header")
    client.post("/cluster/llm", json={"idea_id": id1})

    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Search Feature", is_new=False
    )
    id2 = add_idea(client, "Include a search feature in the top navigation")
    client.post("/cluster/llm", json={"idea_id": id2})

    # Create one idea in a different cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Recommendations", is_new=True
    )
    id3 = add_idea(client, "Build a recommendation engine for product suggestions")
    client.post("/cluster/llm", json={"idea_id": id3})

    resp = client.get("/clusters/llm")
    assert resp.status_code == 200
    clusters = resp.json()
    assert len(clusters) == 2

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


@patch("cluster_api.engines.llm_engine._get_client")
def test_get_llm_clusters_empty(mock_get_client, client):
    """GET /clusters/llm with no clusters should return an empty list."""
    resp = client.get("/clusters/llm")
    assert resp.status_code == 200
    assert resp.json() == []


@patch("cluster_api.engines.llm_engine._get_client")
def test_cluster_multiple_ideas_incrementally(mock_get_client, client):
    """Multiple similar ideas should be assigned to the same cluster incrementally."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    ideas = [
        "Add email notifications when user accounts are modified",
        "Send email notifications when user accounts are updated",
        "Notify users via email when their account is changed",
    ]

    idea_ids = [add_idea(client, text) for text in ideas]

    # First idea creates a new cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Account Email Notifications", is_new=True
    )
    r1 = client.post("/cluster/llm", json={"idea_id": idea_ids[0]})
    assert r1.status_code == 200
    assert r1.json()["is_new"] is True

    # Subsequent ideas assigned to existing cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Account Email Notifications", is_new=False
    )
    r2 = client.post("/cluster/llm", json={"idea_id": idea_ids[1]})
    assert r2.status_code == 200
    assert r2.json()["is_new"] is False

    r3 = client.post("/cluster/llm", json={"idea_id": idea_ids[2]})
    assert r3.status_code == 200
    assert r3.json()["is_new"] is False

    # All should be in the same cluster
    assert r1.json()["cluster_id"] == r2.json()["cluster_id"] == r3.json()["cluster_id"]


@patch("cluster_api.engines.llm_engine._get_client")
def test_llm_fallback_when_cluster_not_found(mock_get_client, client):
    """If LLM says is_new=False but the cluster name doesn't match, create a new one."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # LLM says to assign to a non-existent cluster
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Nonexistent Cluster", is_new=False
    )
    idea_id = add_idea(client, "Some idea that the LLM misclassifies")
    resp = client.post("/cluster/llm", json={"idea_id": idea_id})
    assert resp.status_code == 200
    data = resp.json()
    # Should fall back to creating a new cluster
    assert data["is_new"] is True
    assert data["cluster_name"] == "Nonexistent Cluster"


@patch("cluster_api.engines.llm_engine._get_client")
def test_cluster_duplicate_idea_rejected(mock_get_client, client):
    """Clustering the same idea twice with LLM should return 409."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "Test Cluster", is_new=True
    )
    mock_get_client.return_value = mock_client

    idea_id = add_idea(client, "Improve user onboarding")
    resp = client.post("/cluster/llm", json={"idea_id": idea_id})
    assert resp.status_code == 200

    # Second attempt should be rejected
    resp = client.post("/cluster/llm", json={"idea_id": idea_id})
    assert resp.status_code == 409
