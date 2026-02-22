from cluster_api.db import Idea, Cluster, IdeaCluster, get_session
from cluster_api.models import AddIdeaRequest, AddIdeaResponse, ClusterResponse, IdeaOut


def test_create_idea(db_session):
    idea = Idea(text="Build a better mousetrap", user_id="user1")
    db_session.add(idea)
    db_session.commit()
    db_session.refresh(idea)

    assert idea.id is not None
    assert idea.text == "Build a better mousetrap"
    assert idea.user_id == "user1"
    assert idea.created_at is not None


def test_create_cluster_and_assign_idea(db_session):
    idea = Idea(text="Solar-powered car", user_id="user2")
    db_session.add(idea)
    db_session.commit()

    cluster = Cluster(name="Green Energy", method="bertopic")
    db_session.add(cluster)
    db_session.commit()

    assignment = IdeaCluster(idea_id=idea.id, cluster_id=cluster.id)
    db_session.add(assignment)
    db_session.commit()

    # Verify the relationship
    db_session.refresh(idea)
    db_session.refresh(cluster)
    assert len(idea.clusters) == 1
    assert idea.clusters[0].cluster.name == "Green Energy"
    assert len(cluster.ideas) == 1
    assert cluster.ideas[0].idea.text == "Solar-powered car"


def test_pydantic_add_idea_request():
    req = AddIdeaRequest(text="A new idea", user_id="user1")
    assert req.text == "A new idea"
    assert req.user_id == "user1"


def test_pydantic_add_idea_response():
    resp = AddIdeaResponse(idea_id=42)
    assert resp.idea_id == 42


def test_pydantic_cluster_response():
    resp = ClusterResponse(
        cluster_id=1,
        name="Tech Ideas",
        ideas=[IdeaOut(id=1, text="AI assistant", user_id="user1")],
    )
    assert resp.cluster_id == 1
    assert resp.name == "Tech Ideas"
    assert len(resp.ideas) == 1
    assert resp.ideas[0].text == "AI assistant"


def test_multiple_ideas_in_cluster(db_session):
    ideas = [
        Idea(text="Wind turbine design", user_id="user1"),
        Idea(text="Hydroelectric dam improvements", user_id="user2"),
        Idea(text="Geothermal energy extraction", user_id="user3"),
    ]
    for idea in ideas:
        db_session.add(idea)
    db_session.commit()

    cluster = Cluster(name="Renewable Energy", method="llm")
    db_session.add(cluster)
    db_session.commit()

    for idea in ideas:
        db_session.add(IdeaCluster(idea_id=idea.id, cluster_id=cluster.id))
    db_session.commit()

    db_session.refresh(cluster)
    assert len(cluster.ideas) == 3


def test_idea_in_multiple_clusters(db_session):
    idea = Idea(text="Electric vehicle battery", user_id="user1")
    db_session.add(idea)
    db_session.commit()

    cluster1 = Cluster(name="Green Energy", method="bertopic")
    cluster2 = Cluster(name="Transportation", method="llm")
    db_session.add(cluster1)
    db_session.add(cluster2)
    db_session.commit()

    db_session.add(IdeaCluster(idea_id=idea.id, cluster_id=cluster1.id))
    db_session.add(IdeaCluster(idea_id=idea.id, cluster_id=cluster2.id))
    db_session.commit()

    db_session.refresh(idea)
    assert len(idea.clusters) == 2
