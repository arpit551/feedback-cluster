from contextlib import asynccontextmanager

from fastapi import FastAPI

from cluster_api.config import settings
import cluster_api.db as db_module
from cluster_api.db import Idea, get_session, init_db
from cluster_api.models import AddIdeaRequest, AddIdeaResponse, IdeaOut


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
