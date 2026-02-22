import json

from openai import OpenAI

from sqlalchemy import func

from cluster_api.config import settings
from cluster_api.db import Cluster, Idea, IdeaCluster, get_session


def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def cluster_idea(idea_id: int) -> dict:
    """Cluster an idea using LLM-based classification.

    Returns dict with cluster_id, cluster_name, and is_new.
    """
    session = get_session()
    try:
        idea = session.query(Idea).filter(Idea.id == idea_id).first()
        if idea is None:
            raise ValueError(f"Idea {idea_id} not found")
        idea_text = idea.text
    finally:
        session.close()

    # Load existing LLM clusters
    session = get_session()
    try:
        existing_clusters = (
            session.query(Cluster).filter(Cluster.method == "llm").all()
        )
        cluster_data = [(c.id, c.name) for c in existing_clusters]
    finally:
        session.close()

    # Build prompt for the LLM -- separate system instructions from user content
    if cluster_data:
        cluster_list = "\n".join(f"- {name}" for _, name in cluster_data)
        system_msg = (
            "You are a topic classifier. Given a new idea and a list of existing clusters, "
            "decide whether the idea fits an existing cluster or needs a new one. "
            "If the idea fits an existing cluster, return its exact name. "
            "If it needs a new cluster, suggest a short descriptive name for the new cluster."
        )
        user_msg = f"Existing clusters:\n{cluster_list}\n\nNew idea: \"{idea_text}\""
    else:
        system_msg = (
            "You are a topic classifier. There are no existing clusters yet. "
            "Suggest a short descriptive name for a new cluster that the given idea belongs to."
        )
        user_msg = f"New idea: \"{idea_text}\""

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "cluster_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "cluster_name": {
                                "type": "string",
                                "description": "The name of the cluster (existing or new)",
                            },
                            "is_new": {
                                "type": "boolean",
                                "description": "True if this is a new cluster, False if existing",
                            },
                        },
                        "required": ["cluster_name", "is_new"],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned empty response")

    try:
        decision = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response: {e}") from e

    cluster_name = decision["cluster_name"]
    is_new = decision["is_new"]

    # If the LLM says it fits an existing cluster, find it
    session = get_session()
    try:
        if not is_new:
            # Look up the existing cluster by name
            existing = (
                session.query(Cluster)
                .filter(Cluster.method == "llm", func.lower(Cluster.name) == cluster_name.lower())
                .first()
            )
            if existing is not None:
                assignment = IdeaCluster(idea_id=idea_id, cluster_id=existing.id)
                session.add(assignment)
                session.commit()
                return {
                    "cluster_id": existing.id,
                    "cluster_name": existing.name,
                    "is_new": False,
                }
            # If the LLM referenced a cluster that doesn't exist, treat as new
            is_new = True

        # Create new cluster
        new_cluster = Cluster(name=cluster_name, method="llm")
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
    finally:
        session.close()
