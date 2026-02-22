import json

from openai import OpenAI

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

    # Build prompt for the LLM
    if cluster_data:
        cluster_list = "\n".join(f"- {name}" for _, name in cluster_data)
        prompt = (
            f"You are a topic classifier. Given a new idea and a list of existing clusters, "
            f"decide whether the idea fits an existing cluster or needs a new one.\n\n"
            f"Existing clusters:\n{cluster_list}\n\n"
            f"New idea: \"{idea_text}\"\n\n"
            f"If the idea fits an existing cluster, return its exact name. "
            f"If it needs a new cluster, suggest a short descriptive name for the new cluster."
        )
    else:
        prompt = (
            f"You are a topic classifier. There are no existing clusters yet.\n\n"
            f"New idea: \"{idea_text}\"\n\n"
            f"Suggest a short descriptive name for a new cluster that this idea belongs to."
        )

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
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
                            "description": "True if this is a new cluster, False if assigning to an existing one",
                        },
                    },
                    "required": ["cluster_name", "is_new"],
                    "additionalProperties": False,
                },
            },
        },
    )

    decision = json.loads(response.choices[0].message.content)
    cluster_name = decision["cluster_name"]
    is_new = decision["is_new"]

    # If the LLM says it fits an existing cluster, find it
    session = get_session()
    try:
        if not is_new:
            # Look up the existing cluster by name
            existing = (
                session.query(Cluster)
                .filter(Cluster.method == "llm", Cluster.name == cluster_name)
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
