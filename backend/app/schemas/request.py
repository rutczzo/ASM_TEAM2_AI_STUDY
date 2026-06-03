from pydantic import BaseModel


class RecommendRequest(BaseModel):
    user_input: str