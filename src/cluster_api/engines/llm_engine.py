import json

from openai import OpenAI

from sqlalchemy import func

from cluster_api.config import settings
from cluster_api.db import Cluster, Idea, IdeaCluster, get_session
from cluster_api.exceptions import AlreadyClusteredError, IdeaNotFoundError


def _get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "LLM clustering requires a valid OpenAI API key."
        )
    return OpenAI(api_key=settings.openai_api_key)


def cluster_idea(idea_id: int) -> dict:
    """Cluster an idea using LLM-based classification.

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
    system_msg = (
        "You are a strict topic classifier for user-submitted product ideas.\n\n"
        "RULES:\n"
        "1. Only assign an idea to an existing cluster if it is CLEARLY about the same "
        "specific topic. When in doubt, create a new cluster.\n"
        "2. Cluster names MUST be 2-5 words, specific and descriptive "
        '(e.g. "Dark Mode Toggle", "Email Account Alerts", "Search Navigation Bar"). '
        "Avoid generic names like \"Feature Request\" or \"Improvement\".\n"
        "3. Do NOT over-group. Two ideas that share a vague theme but discuss different "
        "features must go in separate clusters.\n"
        "4. If there are no existing clusters, create a new one."
    )

    if cluster_data:
        # Include cluster names with representative idea examples
        session2 = get_session()
        try:
            cluster_info_parts = []
            for cid, cname in cluster_data:
                assignments = (
                    session2.query(IdeaCluster)
                    .filter(IdeaCluster.cluster_id == cid)
                    .all()
                )
                count = len(assignments)
                idea_ids = [a.idea_id for a in assignments]
                # Get up to 3 representative ideas
                sample_ideas = (
                    session2.query(Idea)
                    .filter(Idea.id.in_(idea_ids[:3]))
                    .all()
                )
                examples = [f'  "{idea.text}"' for idea in sample_ideas]
                examples_str = "\n".join(examples)
                cluster_info_parts.append(
                    f"- {cname} ({count} ideas)\n  Examples:\n{examples_str}"
                )
            cluster_list = "\n".join(cluster_info_parts)
        finally:
            session2.close()

        user_msg = (
            f"Existing clusters:\n{cluster_list}\n\n"
            f'New idea: "{idea_text}"\n\n'
            "Remember: only assign to an existing cluster if the idea is clearly about "
            "the SAME specific topic. Do not force ideas into existing clusters."
        )
    else:
        user_msg = f'New idea: "{idea_text}"'

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

    # Normalize cluster name: strip whitespace, reject blank names
    cluster_name = cluster_name.strip()
    if not cluster_name:
        raise RuntimeError("LLM returned an empty cluster name")

    # If the LLM says it fits an existing cluster, find it
    session = get_session()
    try:
        existing_assignment = (
            session.query(IdeaCluster)
            .join(Cluster)
            .filter(IdeaCluster.idea_id == idea_id, Cluster.method == "llm")
            .first()
        )
        if existing_assignment is not None:
            raise AlreadyClusteredError(idea_id, "llm")

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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
