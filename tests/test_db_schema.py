"""Tests for DB schema: embedding, centroid, and size columns."""

import numpy as np

from cluster_api.db import Cluster, Idea


def test_idea_embedding_column(db_session):
    """Idea.embedding stores and retrieves numpy arrays as bytes."""
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    idea = Idea(text="test idea", user_id="u1", embedding=embedding.tobytes())
    db_session.add(idea)
    db_session.commit()
    db_session.refresh(idea)

    loaded = np.frombuffer(idea.embedding, dtype=np.float32)
    np.testing.assert_array_almost_equal(loaded, embedding)


def test_idea_embedding_nullable(db_session):
    """Idea.embedding defaults to None."""
    idea = Idea(text="no embedding", user_id="u1")
    db_session.add(idea)
    db_session.commit()
    db_session.refresh(idea)

    assert idea.embedding is None


def test_cluster_centroid_and_size(db_session):
    """Cluster stores centroid as bytes and size as integer."""
    centroid = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    cluster = Cluster(
        name="Test Cluster",
        method="bertopic",
        centroid=centroid.tobytes(),
        size=3,
    )
    db_session.add(cluster)
    db_session.commit()
    db_session.refresh(cluster)

    loaded = np.frombuffer(cluster.centroid, dtype=np.float32)
    np.testing.assert_array_almost_equal(loaded, centroid)
    assert cluster.size == 3


def test_cluster_centroid_nullable(db_session):
    """Cluster.centroid defaults to None, size defaults to 0."""
    cluster = Cluster(name="Empty", method="bertopic")
    db_session.add(cluster)
    db_session.commit()
    db_session.refresh(cluster)

    assert cluster.centroid is None
    assert cluster.size == 0
