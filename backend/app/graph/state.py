from typing import TypedDict, Any


class GraphState(TypedDict, total=False):
    user_input: str

    parsed_input: dict[str, Any]
    gap_context: dict[str, Any]

    search_query: str
    retrieved_mentors: list[dict[str, Any]]

    evaluated_mentors: list[dict[str, Any]]
    is_recommendation_confident: bool
    refined_query: str
    retry_count: int

    final_recommendations: list[dict[str, Any]]
    message: str
