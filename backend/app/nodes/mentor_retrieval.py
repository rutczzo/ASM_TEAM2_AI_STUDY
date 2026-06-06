from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MENTORS_PATH = Path(__file__).resolve().parents[3] / "data" / "mentors.json"
MATCH_FIELDS = ("keywords", "can_help", "domain", "profile_summary")
FIELD_WEIGHTS = {
    "keywords": 4.0,
    "can_help": 3.0,
    "domain": 2.0,
    "profile_summary": 1.0,
}
STOPWORDS = {
    "and",
    "or",
    "the",
    "for",
    "with",
    "quality",
    "based",
}


def build_search_query(gap_context: dict[str, Any]) -> str:
    values = [
        *gap_context.get("query_hints", []),
        *gap_context.get("needed_mentor_expertise", []),
        *gap_context.get("gap_categories", []),
        gap_context.get("main_gap", ""),
    ]
    query_parts = _unique_display_values(values)
    return " ".join(query_parts[:14])


def load_mentors(path: Path | None = None) -> list[dict[str, Any]]:
    mentors_path = path or DEFAULT_MENTORS_PATH
    if not mentors_path.exists():
        return []

    raw_content = mentors_path.read_text(encoding="utf-8").strip()
    if not raw_content:
        return []

    try:
        mentors = json.loads(raw_content)
    except json.JSONDecodeError:
        return []

    if not isinstance(mentors, list):
        return []
    return [mentor for mentor in mentors if isinstance(mentor, dict)]


def retrieve_mentor_candidates(
    gap_context: dict[str, Any],
    mentors: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not gap_context or not mentors or limit <= 0:
        return []

    gap_terms = _extract_gap_terms(gap_context)
    scored_candidates = []
    for index, mentor in enumerate(mentors):
        score, matched_keywords, matched_fields = _score_mentor(mentor, gap_terms)
        if score <= 0:
            continue

        scored_candidates.append(
            (
                -score,
                index,
                {
                    **mentor,
                    "retrieval_score": round(score, 2),
                    "matched_keywords": matched_keywords,
                    "matched_fields": matched_fields,
                },
            )
        )

    scored_candidates.sort(key=lambda item: (item[0], item[1]))
    return [candidate for _, _, candidate in scored_candidates[:limit]]


def mentor_retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    gap_context = state.get("gap_context")
    if not gap_context:
        return {"search_query": "", "retrieved_mentors": []}

    mentors = load_mentors()
    return {
        "search_query": build_search_query(gap_context),
        "retrieved_mentors": retrieve_mentor_candidates(gap_context, mentors),
    }


def _score_mentor(
    mentor: dict[str, Any],
    gap_terms: set[str],
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    matched_keywords: list[str] = []
    matched_fields: list[str] = []

    for field_name in MATCH_FIELDS:
        field_values = _field_values(mentor, field_name)
        field_matches = _matched_terms(field_values, gap_terms)
        if not field_matches:
            continue

        matched_fields.append(field_name)
        score += FIELD_WEIGHTS[field_name] * len(field_matches)
        matched_keywords.extend(_field_match_labels(field_name, field_values, gap_terms))

    return score, _unique_display_values(matched_keywords), matched_fields


def _extract_gap_terms(gap_context: dict[str, Any]) -> set[str]:
    values = [
        gap_context.get("main_gap", ""),
        *gap_context.get("gap_categories", []),
        *gap_context.get("needed_mentor_expertise", []),
        *gap_context.get("query_hints", []),
    ]
    terms: set[str] = set()
    for value in values:
        terms.update(_terms_for_matching(str(value)))
    return terms


def _matched_terms(values: list[str], gap_terms: set[str]) -> set[str]:
    matched: set[str] = set()
    for value in values:
        value_terms = _terms_for_matching(value)
        for value_term in value_terms:
            for gap_term in gap_terms:
                if _terms_match(value_term, gap_term):
                    matched.add(value_term)
    return matched


def _field_match_labels(
    field_name: str,
    field_values: list[str],
    gap_terms: set[str],
) -> list[str]:
    labels: list[str] = []
    for value in field_values:
        value_terms = _terms_for_matching(value)
        if any(
            _terms_match(value_term, gap_term)
            for value_term in value_terms
            for gap_term in gap_terms
        ):
            labels.append(_normalize_label(value))

    if field_name == "keywords":
        return labels
    return [label for label in labels if len(label) > 1]


def _terms_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) >= 3 and left in right:
        return True
    return len(right) >= 3 and right in left


def _terms_for_matching(value: str) -> set[str]:
    normalized = _normalize_label(value)
    if not normalized:
        return set()

    terms = {normalized}
    terms.update(_split_terms(normalized))
    return {term for term in terms if term and term not in STOPWORDS}


def _split_terms(value: str) -> set[str]:
    raw_terms = re.split(r"[^0-9a-zA-Z가-힣+#/.-]+", value)
    return {
        term
        for term in raw_terms
        if len(term) >= 2 and term not in STOPWORDS
    }


def _field_values(mentor: dict[str, Any], field_name: str) -> list[str]:
    value = mentor.get(field_name, [])
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _unique_display_values(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = _normalize_label(text)
        if not text or key in seen:
            continue
        result.append(text)
        seen.add(key)
    return result
