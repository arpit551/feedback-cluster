from pydantic import BaseModel, Field


class AddIdeaRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    user_id: str = Field(..., min_length=1, max_length=255)


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
