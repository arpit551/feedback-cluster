from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from cluster_api.config import settings
from cluster_api.db import Cluster, Idea, IdeaCluster, get_session
from cluster_api.exceptions import AlreadyClusteredError, IdeaNotFoundError

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _compute_embedding(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode([text])[0]


def _get_cluster_centroid(cluster_id: int) -> Optional[np.ndarray]:
    """Compute centroid of a cluster by averaging embeddings of its ideas."""
    session = get_session()
    try:
        assignments = (
            session.query(IdeaCluster).filter(IdeaCluster.cluster_id == cluster_id).all()
        )
        if not assignments:
            return None
        idea_ids = [a.idea_id for a in assignments]
        ideas = session.query(Idea).filter(Idea.id.in_(idea_ids)).all()
        texts = [idea.text for idea in ideas]
    finally:
        session.close()

    if not texts:
        return None

    model = _get_model()
    embeddings = model.encode(texts)
    return np.mean(embeddings, axis=0)


def cluster_idea(idea_id: int) -> dict:
    """Cluster an idea using embedding similarity.

    Returns dict with cluster_id, cluster_name, and is_new.
    """
    session = get_session()
    try:
        idea = session.query(Idea).filter(Idea.id == idea_id).first()
        if idea is None:
            raise IdeaNotFoundError(idea_id)
        idea_text = idea.text
    finally:
        session.close()

    idea_embedding = _compute_embedding(idea_text)

    # Load existing bertopic clusters
    session = get_session()
    try:
        existing_clusters = (
            session.query(Cluster).filter(Cluster.method == "bertopic").all()
        )
        cluster_data = [(c.id, c.name) for c in existing_clusters]
    finally:
        session.close()

    best_cluster_id = None
    best_cluster_name = None
    best_similarity = -1.0

    for cid, cname in cluster_data:
        centroid = _get_cluster_centroid(cid)
        if centroid is not None:
            sim = cosine_similarity(
                idea_embedding.reshape(1, -1), centroid.reshape(1, -1)
            )[0][0]
            if sim > best_similarity:
                best_similarity = sim
                best_cluster_id = cid
                best_cluster_name = cname

    session = get_session()
    try:
        existing = (
            session.query(IdeaCluster)
            .join(Cluster)
            .filter(IdeaCluster.idea_id == idea_id, Cluster.method == "bertopic")
            .first()
        )
        if existing is not None:
            raise AlreadyClusteredError(idea_id, "bertopic")

        if best_cluster_id is not None and best_similarity >= settings.similarity_threshold:
            # Assign to existing cluster
            assignment = IdeaCluster(idea_id=idea_id, cluster_id=best_cluster_id)
            session.add(assignment)
            session.commit()
            return {
                "cluster_id": best_cluster_id,
                "cluster_name": best_cluster_name,
                "is_new": False,
            }
        else:
            # Create new cluster
            # Use a truncated version of the idea text as the cluster name
            name = idea_text[:80] if len(idea_text) > 80 else idea_text
            new_cluster = Cluster(name=name, method="bertopic")
            session.add(new_cluster)
            session.flush()

            assignment = IdeaCluster(idea_id=idea_id, cluster_id=new_cluster.id)
            session.add(assignment)
            session.commit()
            return {
                "cluster_id": new_cluster.id,
                "cluster_name": new_cluster.name,
                "is_new": True,
            }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
