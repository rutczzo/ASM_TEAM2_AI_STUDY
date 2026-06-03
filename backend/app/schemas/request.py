from pydantic import BaseModel


class RecommendRequest(BaseModel):
    project_text: str
    tech_stack: list[str] = []
    stage: str = ""
    clarify_answer: str | None = None