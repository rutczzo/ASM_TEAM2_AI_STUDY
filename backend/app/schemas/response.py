from pydantic import BaseModel
from typing import List


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