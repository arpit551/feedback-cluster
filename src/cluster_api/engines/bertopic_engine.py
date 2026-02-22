import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from cluster_api.config import settings
from cluster_api.db import Cluster, Idea, IdeaCluster, get_session
from cluster_api.exceptions import AlreadyClusteredError, IdeaNotFoundError

logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Embedding clustering requires a valid OpenAI API key."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _compute_embedding(text: str) -> np.ndarray:
    """Compute embedding for a text using OpenAI embeddings API."""
    client = _get_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=[text],
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _generate_cluster_name(texts: list) -> str:
    """Generate a short 2-4 word topic label using the LLM."""
    client = _get_client()
    sample = texts[:5]
    ideas_str = "\n".join(f'- "{t}"' for t in sample)
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a 2-4 word topic label for the following ideas. "
                        "Respond with just the label, nothing else."
                    ),
                },
                {"role": "user", "content": ideas_str},
            ],
        )
        name = response.choices[0].message.content
        if name is None:
            return "Misc"
        return name.strip().strip('"')
    except Exception:
        logger.exception("Failed to generate cluster name via LLM")
        return texts[0][:80] if texts else "Misc"


def _get_or_compute_embedding(idea_id: int, idea_text: str, stored: Optional[bytes]) -> np.ndarray:
    """Get stored embedding or compute and store it."""
    if stored is not None:
        return np.frombuffer(stored, dtype=np.float32).copy()

    embedding = _compute_embedding(idea_text)
    session = get_session()
    try:
        db_idea = session.query(Idea).filter(Idea.id == idea_id).first()
        if db_idea:
            db_idea.embedding = embedding.tobytes()
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return embedding


def cluster_idea(idea_id: int) -> dict:
    """Cluster an idea using embedding cosine similarity.

    Returns dict with cluster_id, cluster_name, is_new, and confidence.
    """
    session = get_session()
    try:
        idea = session.query(Idea).filter(Idea.id == idea_id).first()
        if idea is None:
            raise IdeaNotFoundError(idea_id)

        existing = (
            session.query(IdeaCluster)
            .join(Cluster)
            .filter(IdeaCluster.idea_id == idea_id, Cluster.method == "bertopic")
            .first()
        )
        if existing is not None:
            raise AlreadyClusteredError(idea_id, "bertopic")

        idea_text = idea.text
        idea_embedding = _get_or_compute_embedding(idea_id, idea.text, idea.embedding)
    finally:
        session.close()

    # Load existing bertopic clusters with centroids
    session = get_session()
    try:
        existing_clusters = (
            session.query(Cluster)
            .filter(Cluster.method == "bertopic", Cluster.centroid.isnot(None))
            .all()
        )
        cluster_data = [
            (c.id, c.name, np.frombuffer(c.centroid, dtype=np.float32).copy(), c.size)
            for c in existing_clusters
        ]
    finally:
        session.close()

    best_cluster_id = None
    best_cluster_name = None
    best_similarity = -1.0
    best_centroid = None
    best_size = 0

    for cid, cname, centroid, size in cluster_data:
        sim = _cosine_similarity(idea_embedding, centroid)
        if sim > best_similarity:
            best_similarity = sim
            best_cluster_id = cid
            best_cluster_name = cname
            best_centroid = centroid
            best_size = size

    session = get_session()
    try:
        if best_cluster_id is not None and best_similarity >= settings.similarity_threshold:
            # Assign to existing cluster and update centroid incrementally
            assignment = IdeaCluster(idea_id=idea_id, cluster_id=best_cluster_id)
            session.add(assignment)

            cluster = session.query(Cluster).filter(Cluster.id == best_cluster_id).first()
            new_centroid = (best_centroid * best_size + idea_embedding) / (best_size + 1)
            cluster.centroid = new_centroid.tobytes()
            cluster.size = best_size + 1

            session.commit()
            return {
                "cluster_id": best_cluster_id,
                "cluster_name": best_cluster_name,
                "is_new": False,
                "confidence": float(best_similarity),
            }
        else:
            # Create new cluster with LLM-generated name
            name = _generate_cluster_name([idea_text])
            new_cluster = Cluster(
                name=name,
                method="bertopic",
                centroid=idea_embedding.tobytes(),
                size=1,
            )
            session.add(new_cluster)
            session.flush()

            assignment = IdeaCluster(idea_id=idea_id, cluster_id=new_cluster.id)
            session.add(assignment)
            session.commit()
            return {
                "cluster_id": new_cluster.id,
                "cluster_name": new_cluster.name,
                "is_new": True,
                "confidence": 1.0,
            }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


