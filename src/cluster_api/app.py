from contextlib import asynccontextmanager

from fastapi import FastAPI

from fastapi import HTTPException
from pydantic import BaseModel

from cluster_api.config import settings
import cluster_api.db as db_module
from cluster_api.db import Cluster, Idea, IdeaCluster, get_session, init_db
from cluster_api.engines.bertopic_engine import cluster_idea as bertopic_cluster_idea
from cluster_api.engines.llm_engine import cluster_idea as llm_cluster_idea
from cluster_api.models import AddIdeaRequest, AddIdeaResponse, ClusterResponse, IdeaOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    if db_module._engine is None:
        init_db(settings.db_path)
    yield


app = FastAPI(title="Idea Clustering API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ideas", response_model=AddIdeaResponse, status_code=201)
def add_idea(req: AddIdeaRequest):
    session = get_session()
    try:
        idea = Idea(text=req.text, user_id=req.user_id)
        session.add(idea)
        session.commit()
        session.refresh(idea)
        return AddIdeaResponse(idea_id=idea.id)
    finally:
        session.close()


@app.get("/ideas", response_model=list[IdeaOut])
def list_ideas():
    session = get_session()
    try:
        ideas = session.query(Idea).all()
        return [IdeaOut(id=i.id, text=i.text, user_id=i.user_id) for i in ideas]
    finally:
        session.close()


class ClusterIdeaRequest(BaseModel):
    idea_id: int


@app.post("/cluster/bertopic")
def cluster_bertopic(req: ClusterIdeaRequest):
    session = get_session()
    try:
        idea = session.query(Idea).filter(Idea.id == req.idea_id).first()
        if idea is None:
            raise HTTPException(status_code=404, detail="Idea not found")
    finally:
        session.close()

    result = bertopic_cluster_idea(req.idea_id)
    return result


@app.get("/clusters/bertopic", response_model=list[ClusterResponse])
def list_bertopic_clusters():
    session = get_session()
    try:
        clusters = session.query(Cluster).filter(Cluster.method == "bertopic").all()
        result = []
        for cluster in clusters:
            assignments = (
                session.query(IdeaCluster)
                .filter(IdeaCluster.cluster_id == cluster.id)
                .all()
            )
            idea_ids = [a.idea_id for a in assignments]
            ideas = session.query(Idea).filter(Idea.id.in_(idea_ids)).all() if idea_ids else []
            result.append(
                ClusterResponse(
                    cluster_id=cluster.id,
                    name=cluster.name,
                    ideas=[IdeaOut(id=i.id, text=i.text, user_id=i.user_id) for i in ideas],
                )
            )
        return result
    finally:
        session.close()


@app.post("/cluster/llm")
def cluster_llm(req: ClusterIdeaRequest):
    session = get_session()
    try:
        idea = session.query(Idea).filter(Idea.id == req.idea_id).first()
        if idea is None:
            raise HTTPException(status_code=404, detail="Idea not found")
    finally:
        session.close()

    result = llm_cluster_idea(req.idea_id)
    return result


@app.get("/clusters/llm", response_model=list[ClusterResponse])
def list_llm_clusters():
    session = get_session()
    try:
        clusters = session.query(Cluster).filter(Cluster.method == "llm").all()
        result = []
        for cluster in clusters:
            assignments = (
                session.query(IdeaCluster)
                .filter(IdeaCluster.cluster_id == cluster.id)
                .all()
            )
            idea_ids = [a.idea_id for a in assignments]
            ideas = session.query(Idea).filter(Idea.id.in_(idea_ids)).all() if idea_ids else []
            result.append(
                ClusterResponse(
                    cluster_id=cluster.id,
                    name=cluster.name,
                    ideas=[IdeaOut(id=i.id, text=i.text, user_id=i.user_id) for i in ideas],
                )
            )
        return result
    finally:
        session.close()
