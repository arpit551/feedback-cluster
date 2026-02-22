def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_idea(client):
    resp = client.post("/ideas", json={"text": "Use AI to sort feedback", "user_id": "user1"})
    assert resp.status_code == 201
    data = resp.json()
    assert "idea_id" in data
    assert isinstance(data["idea_id"], int)


def test_get_ideas_empty(client):
    resp = client.get("/ideas")
    assert resp.status_code == 200
    assert resp.json() == []


def test_post_and_get_ideas(client):
    client.post("/ideas", json={"text": "Idea A", "user_id": "u1"})
    client.post("/ideas", json={"text": "Idea B", "user_id": "u2"})

    resp = client.get("/ideas")
    assert resp.status_code == 200
    ideas = resp.json()
    assert len(ideas) == 2
    assert ideas[0]["text"] == "Idea A"
    assert ideas[0]["user_id"] == "u1"
    assert ideas[1]["text"] == "Idea B"
    assert ideas[1]["user_id"] == "u2"


def test_post_idea_missing_fields(client):
    resp = client.post("/ideas", json={"text": "No user id"})
    assert resp.status_code == 422

    resp = client.post("/ideas", json={"user_id": "u1"})
    assert resp.status_code == 422


def test_idea_ids_are_unique(client):
    r1 = client.post("/ideas", json={"text": "First", "user_id": "u1"})
    r2 = client.post("/ideas", json={"text": "Second", "user_id": "u1"})
    assert r1.json()["idea_id"] != r2.json()["idea_id"]
