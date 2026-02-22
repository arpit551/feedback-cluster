from pydantic import BaseModel, Field, field_validator


class AddIdeaRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    user_id: str = Field(..., min_length=1, max_length=255)

    @field_validator("text", "user_id")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


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
