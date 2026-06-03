from typing import List, Literal
from pydantic import BaseModel


class MentorRecommendation(BaseModel):
    name: str
    domain: List[str]
    score: float
    reason: str
    matched_keywords: List[str]


class RecommendResponse(BaseModel):
    status: str
    message: str
    recommendations: List[MentorRecommendation]


class ClarificationResponse(BaseModel):
    status: Literal["need_clarification"]
    question: str
    options: List[str]
