from pydantic import BaseModel


class AddIdeaRequest(BaseModel):
    text: str
    user_id: str


class AddIdeaResponse(BaseModel):
    idea_id: int


class IdeaOut(BaseModel):
    id: int
    text: str
    user_id: str


class ClusterResponse(BaseModel):
    cluster_id: int
    name: str
    ideas: list[IdeaOut]
