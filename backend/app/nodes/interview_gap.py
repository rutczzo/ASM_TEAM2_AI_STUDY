import re
from dataclasses import dataclass

from backend.app.schemas.gap import GapContext, ParsedInput


@dataclass(frozen=True)
class GapRule:
    categories: tuple[str, ...]
    signals: tuple[str, ...]
    expertise: tuple[str, ...]
    query_hints: tuple[str, ...]


GAP_RULES: tuple[GapRule, ...] = (
    GapRule(
        categories=("RAG", "Embedding", "Retrieval"),
        signals=(
            "rag",
            "검색",
            "검색 품질",
            "검색 정확도",
            "임베딩",
            "embedding",
            "벡터",
            "vector",
            "유사도",
            "chunk",
            "청크",
            "retrieval",
        ),
        expertise=("RAG 검색 구조 설계", "임베딩 전략", "벡터 DB 활용"),
        query_hints=(
            "RAG",
            "retrieval quality",
            "embedding strategy",
            "vector database",
            "semantic search",
        ),
    ),
    GapRule(
        categories=("Agentic Workflow", "LangGraph", "LLM Orchestration"),
        signals=(
            "langgraph",
            "agent",
            "에이전트",
            "분기",
            "상태",
            "state",
            "workflow",
            "워크플로우",
            "orchestration",
        ),
        expertise=("LangGraph 구조 설계", "상태 기반 워크플로우 설계", "에이전트 분기 설계"),
        query_hints=(
            "LangGraph",
            "agent workflow",
            "LLM orchestration",
            "state management",
        ),
    ),
    GapRule(
        categories=("LLM Evaluation", "Prompt Engineering", "Recommendation"),
        signals=(
            "추천",
            "근거",
            "평가",
            "환각",
            "hallucination",
            "신뢰도",
            "적합도",
            "ranking",
            "rank",
            "프롬프트",
            "prompt",
        ),
        expertise=("LLM 평가", "추천 근거 생성", "프롬프트 설계"),
        query_hints=(
            "LLM evaluation",
            "recommendation system",
            "recommendation reasoning",
            "ranking",
            "hallucination mitigation",
            "prompt engineering",
        ),
    ),
    GapRule(
        categories=("Infra", "MLOps", "DevOps"),
        signals=(
            "배포",
            "운영",
            "서버",
            "docker",
            "도커",
            "서빙",
            "serving",
            "api",
            "latency",
            "지연",
            "인프라",
            "infra",
            "mlops",
        ),
        expertise=("배포 구조 설계", "모델 서빙", "운영 환경 구성"),
        query_hints=("deployment", "MLOps", "Docker", "model serving", "infrastructure"),
    ),
    GapRule(
        categories=("Architecture", "Backend", "Scalability"),
        signals=(
            "구조",
            "확장",
            "확장성",
            "백엔드",
            "backend",
            "api",
            "모듈",
            "성능",
            "architecture",
            "system design",
        ),
        expertise=("백엔드 아키텍처 리뷰", "API 설계", "확장성 검토"),
        query_hints=("backend architecture", "API design", "scalability", "system design"),
    ),
    GapRule(
        categories=("Product", "UX", "Research"),
        signals=(
            "사용자",
            "검증",
            "문제 정의",
            "ux",
            "화면",
            "기획",
            "데모",
            "product",
            "research",
        ),
        expertise=("사용자 검증", "문제 정의", "서비스 기획"),
        query_hints=("product strategy", "UX research", "user validation", "service design"),
    ),
)

CORE_QUALITY_CATEGORIES = {
    "RAG",
    "Retrieval",
    "Agentic Workflow",
    "LangGraph",
    "LLM Evaluation",
    "Recommendation",
}

RISK_SIGNALS = (
    "짧",
    "기간",
    "데모",
    "로컬",
    "합성",
    "제약",
    "부족",
    "초기",
    "마감",
)


def analyze_project_gap(parsed_input: ParsedInput | dict) -> GapContext:
    parsed = _to_parsed_input(parsed_input)
    scores = _score_rules(parsed)
    selected_rules = _select_rules(scores, parsed)

    if not selected_rules:
        selected_rules = _fallback_rules(parsed)

    gap_categories = _collect_ranked_values(
        selected_rules, "categories", limit=6
    )
    needed_mentor_expertise = _collect_ranked_values(
        selected_rules, "expertise", limit=4
    )
    query_hints = _unique(
        hint for rule in selected_rules for hint in rule.query_hints
    )[:12]
    matched_fields = _matched_source_fields(parsed, selected_rules)
    source_fields = list(matched_fields.keys())
    priority = _decide_priority(parsed, gap_categories, scores)

    return GapContext(
        main_gap=_build_main_gap(gap_categories, needed_mentor_expertise),
        gap_categories=gap_categories,
        needed_mentor_expertise=needed_mentor_expertise,
        priority=priority,
        reason=_build_reason(gap_categories, matched_fields),
        query_hints=query_hints,
        source_fields=source_fields,
    )


def interview_gap_node(state: dict) -> dict:
    parsed_input = state.get("parsed_input")
    if not _has_analyzable_input(parsed_input):
        return {"gap_context": None}

    gap_context = analyze_project_gap(parsed_input)
    return {"gap_context": gap_context.model_dump()}


def _to_parsed_input(parsed_input: ParsedInput | dict) -> ParsedInput:
    if isinstance(parsed_input, ParsedInput):
        return parsed_input
    return ParsedInput(**parsed_input)


def _has_analyzable_input(parsed_input: ParsedInput | dict | None) -> bool:
    if not parsed_input:
        return False

    parsed = _to_parsed_input(parsed_input)
    return any(
        (
            parsed.project_summary.strip(),
            parsed.concerns,
            parsed.user_goal.strip(),
        )
    )


def _score_rules(parsed: ParsedInput) -> dict[GapRule, int]:
    field_values = {
        "concerns": parsed.concerns,
        "user_goal": [parsed.user_goal],
        "tech_stack": parsed.tech_stack,
        "current_stage": [parsed.current_stage],
        "constraints": parsed.constraints,
        "domain": parsed.domain,
        "project_summary": [parsed.project_summary],
    }
    field_weights = {
        "concerns": 4,
        "user_goal": 1,
        "tech_stack": 2,
        "current_stage": 2,
        "constraints": 2,
        "domain": 1,
        "project_summary": 1,
    }

    scores: dict[GapRule, int] = {}
    for rule in GAP_RULES:
        score = 0
        for field_name, values in field_values.items():
            text = _join_text(values)
            if _contains_signal(text, rule.signals):
                score += field_weights[field_name]
        if score:
            scores[rule] = score
    return scores


def _select_rules(
    scores: dict[GapRule, int], parsed: ParsedInput
) -> tuple[GapRule, ...]:
    eligible_scores = {
        rule: score
        for rule, score in scores.items()
        if score >= 3
        or _first_matching_concern_index(parsed.concerns, rule) < len(parsed.concerns)
    }
    if not eligible_scores:
        eligible_scores = scores

    ranked = sorted(
        eligible_scores.items(),
        key=lambda item: (
            _first_matching_concern_index(parsed.concerns, item[0]),
            -item[1],
        ),
    )
    return tuple(rule for rule, _ in ranked[:3])


def _fallback_rules(parsed: ParsedInput) -> tuple[GapRule, ...]:
    if _has_enough_text_for_fallback(parsed):
        return (GAP_RULES[-1],)
    return ()


def _has_enough_text_for_fallback(parsed: ParsedInput) -> bool:
    text = _join_text(
        [
            parsed.project_summary,
            parsed.user_goal,
            parsed.current_stage,
            *parsed.concerns,
            *parsed.constraints,
            *parsed.domain,
        ]
    )
    return len(text.strip()) >= 20


def _first_matching_concern_index(concerns: list[str], rule: GapRule) -> int:
    for index, concern in enumerate(concerns):
        if _contains_signal(concern, rule.signals):
            return index
    return len(concerns) + 1


def _matched_source_fields(
    parsed: ParsedInput, selected_rules: tuple[GapRule, ...]
) -> dict[str, list[str]]:
    matched_fields: dict[str, list[str]] = {}
    rule_signals = tuple(
        signal for rule in selected_rules for signal in rule.signals
    )
    for field_name in (
        "concerns",
        "tech_stack",
        "current_stage",
        "constraints",
        "user_goal",
        "domain",
        "project_summary",
    ):
        values = _field_values(parsed, field_name)
        matched_values = [
            value for value in values if _contains_signal(value, rule_signals)
        ]
        if matched_values:
            matched_fields[field_name] = matched_values
    for field_name in ("current_stage", "constraints"):
        risk_values = [
            value
            for value in _field_values(parsed, field_name)
            if _contains_signal(value, RISK_SIGNALS)
        ]
        if risk_values:
            matched_fields.setdefault(field_name, [])
            matched_fields[field_name].extend(
                value for value in risk_values if value not in matched_fields[field_name]
            )
    return _sort_matched_fields(matched_fields)


def _decide_priority(
    parsed: ParsedInput, gap_categories: list[str], scores: dict[GapRule, int]
) -> str:
    top_score = max(scores.values(), default=0)
    has_core_quality_gap = any(
        category in CORE_QUALITY_CATEGORIES for category in gap_categories
    )
    has_risk_constraint = _contains_signal(
        _join_text([parsed.current_stage, *parsed.constraints]), RISK_SIGNALS
    )

    if top_score >= 6 and (has_core_quality_gap or has_risk_constraint):
        return "high"
    if top_score >= 3:
        return "medium"
    return "low"


def _build_main_gap(
    gap_categories: list[str], needed_mentor_expertise: list[str]
) -> str:
    if not gap_categories or not needed_mentor_expertise:
        return "프로젝트 약점 분석을 위한 문제 정의 구체화 필요"

    category_text = ", ".join(gap_categories[:2])
    expertise_text = " 및 ".join(needed_mentor_expertise[:2])
    return f"{category_text} 영역의 {expertise_text} 역량 부족"


def _build_reason(
    gap_categories: list[str], matched_fields: dict[str, list[str]]
) -> str:
    field_labels = {
        "concerns": "현재 고민",
        "tech_stack": "기술 스택",
        "current_stage": "진행 단계",
        "constraints": "제약 조건",
        "user_goal": "멘토링 목표",
        "domain": "프로젝트 도메인",
        "project_summary": "프로젝트 요약",
    }
    category_text = ", ".join(gap_categories)
    evidence = _format_evidence(matched_fields, field_labels)

    if evidence:
        return (
            f"{evidence}에서 관련 신호가 확인되어 "
            f"{category_text} 역량이 현재 멘토 검색의 핵심 기준으로 판단됩니다."
        )
    return (
        f"{category_text} 역량 보완이 멘토 검색의 우선 기준으로 판단됩니다."
    )


def _contains_signal(text: str, signals: tuple[str, ...]) -> bool:
    return any(_matches_signal(text, signal) for signal in signals)


def _matches_signal(text: str, signal: str) -> bool:
    if signal.isascii() and signal.replace(" ", "").isalnum():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(signal)}(?![A-Za-z0-9])"
        return re.search(pattern, text, re.IGNORECASE) is not None
    return signal.casefold() in text.casefold()


def _join_text(values: list[str] | tuple[str, ...]) -> str:
    return " ".join(str(value) for value in values if value)


def _field_values(parsed: ParsedInput, field_name: str) -> list[str]:
    value = getattr(parsed, field_name)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _format_evidence(
    matched_fields: dict[str, list[str]], field_labels: dict[str, str]
) -> str:
    evidence_parts = []
    for field_name, values in list(matched_fields.items())[:5]:
        label = field_labels[field_name]
        value_text = ", ".join(values[:3])
        evidence_parts.append(f"{label} '{value_text}'")
    return ", ".join(evidence_parts)


def _sort_matched_fields(matched_fields: dict[str, list[str]]) -> dict[str, list[str]]:
    field_order = (
        "concerns",
        "current_stage",
        "constraints",
        "tech_stack",
        "user_goal",
        "domain",
        "project_summary",
    )
    return {
        field_name: matched_fields[field_name]
        for field_name in field_order
        if field_name in matched_fields
    }


def _unique(values) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _collect_ranked_values(
    selected_rules: tuple[GapRule, ...], field_name: str, limit: int
) -> list[str]:
    primary_values = [
        getattr(rule, field_name)[0]
        for rule in selected_rules
        if getattr(rule, field_name)
    ]
    secondary_values = [
        value
        for rule in selected_rules
        for value in getattr(rule, field_name)[1:]
    ]
    return _unique([*primary_values, *secondary_values])[:limit]
from __future__ import annotations

from typing import Any

from backend.app.graph.state import GraphState


# ─────────────────────────────────────────────────────────────
# Sufficiency check
# Rule-based now; replace _check_sufficiency_rule_based call
# in parse_input with _check_sufficiency_llm when ready.
# ─────────────────────────────────────────────────────────────

_TECH_KEYWORDS: frozenset[str] = frozenset({
    "python", "java", "kotlin", "swift", "go", "rust",
    "react", "vue", "angular", "next", "svelte",
    "node", "express", "django", "fastapi", "spring", "flask", "rails",
    "api", "rest", "graphql", "grpc",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "서버", "백엔드", "프론트", "앱", "웹", "모바일",
    "머신러닝", "딥러닝", "ai", "ml", "llm", "gpt",
    "docker", "kubernetes", "k8s", "aws", "gcp", "azure",
})

_CONCERN_KEYWORDS: tuple[str, ...] = (
    "어렵", "모르", "부족", "고민", "문제", "개선", "도움", "막히",
    "어떻게", "힘들", "필요", "배우", "못", "안 됨", "안됨",
    "이슈", "오류", "에러", "error", "issue",
)


def _check_sufficiency_rule_based(
    project_text: str,
    tech_stack: list[str],
) -> tuple[bool, str, list[str]]:
    """Returns (is_sufficient, question, options)."""
    text_lower = project_text.lower()

    if len(project_text.strip()) < 30 and not tech_stack:
        return (
            False,
            "어떤 프로젝트를 만들고 있는지 조금 더 설명해 주실 수 있나요?",
            ["웹/앱 서비스", "데이터 분석/ML", "API 서버/백엔드", "기타"],
        )

    has_tech = bool(tech_stack) or any(kw in text_lower for kw in _TECH_KEYWORDS)
    has_concern = any(kw in project_text for kw in _CONCERN_KEYWORDS)

    if not has_tech:
        return (
            False,
            "어떤 기술 스택을 사용하고 있나요?",
            ["Python / Django / FastAPI", "JavaScript / React / Node.js", "Java / Spring", "기타 또는 아직 미정"],
        )

    if not has_concern:
        return (
            False,
            "현재 어떤 부분에서 멘토의 도움이 필요한가요?",
            ["설계 및 아키텍처", "특정 기술 구현", "성능 최적화", "프로젝트 방향 및 기획"],
        )

    return True, "", []


def _check_sufficiency_llm(
    project_text: str,
    tech_stack: list[str],
) -> tuple[bool, str, list[str]]:
    """LLM-based sufficiency check. Swap in when LLM integration is ready."""
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# clarify_answer merge
# ─────────────────────────────────────────────────────────────

def _merge_clarify_answer(project_text: str, clarify_answer: str) -> str:
    return f"{project_text}\n추가 정보: {clarify_answer}"


# ─────────────────────────────────────────────────────────────
# Structured parser
# Rule-based now; replace _parse_to_structured_rule_based call
# in parse_input with _parse_to_structured_llm when ready.
# ─────────────────────────────────────────────────────────────

_TECH_EXTRACT_MAP: dict[str, str] = {
    "python": "Python", "java": "Java", "kotlin": "Kotlin",
    "react": "React", "vue": "Vue", "angular": "Angular", "next": "Next.js",
    "node": "Node.js", "django": "Django", "fastapi": "FastAPI",
    "spring": "Spring", "flask": "Flask",
    "docker": "Docker", "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "mysql": "MySQL", "postgresql": "PostgreSQL", "mongodb": "MongoDB",
    "redis": "Redis", "llm": "LLM", "gpt": "GPT",
}

_DOMAIN_MAP: dict[str, str] = {
    "여행": "여행", "일정": "여행", "관광": "여행",
    "의료": "의료/헬스케어", "건강": "의료/헬스케어", "병원": "의료/헬스케어",
    "교육": "교육", "학습": "교육", "강의": "교육",
    "금융": "금융/핀테크", "결제": "금융/핀테크", "투자": "금융/핀테크",
    "쇼핑": "이커머스", "주문": "이커머스", "배달": "이커머스",
    "게임": "게임", "소셜": "소셜/커뮤니티", "커뮤니티": "소셜/커뮤니티",
}

_CONCERN_EXTRACT_KEYWORDS: tuple[str, ...] = (
    "어렵", "모르", "부족", "고민", "문제", "개선", "막히", "힘들",
)

_CONSTRAINT_KEYWORDS: tuple[str, ...] = (
    "1인", "혼자", "마감", "예산", "팀원 없", "기간",
)

_GOAL_KEYWORDS: tuple[str, ...] = (
    "원한다", "바란다", "기대", "필요", "도움", "배우고", "개선하고", "하고 싶",
)


def _parse_to_structured_rule_based(
    project_text: str,
    tech_stack: list[str],
    stage: str,
) -> dict[str, Any]:
    lines = [ln.strip() for ln in project_text.strip().splitlines() if ln.strip()]
    sentences = [s.strip() for s in project_text.replace(".", ".\n").splitlines() if s.strip()]

    project_summary = lines[0][:100] if lines else project_text[:100]

    text_lower = project_text.lower()
    extracted_tech: list[str] = list(tech_stack)
    for kw, label in _TECH_EXTRACT_MAP.items():
        if kw in text_lower and label not in extracted_tech:
            extracted_tech.append(label)

    concerns = [s for s in sentences if any(kw in s for kw in _CONCERN_EXTRACT_KEYWORDS)]

    seen_domains: set[str] = set()
    domains: list[str] = []
    for kw, domain in _DOMAIN_MAP.items():
        if kw in project_text and domain not in seen_domains:
            domains.append(domain)
            seen_domains.add(domain)

    constraints = [s for s in sentences if any(kw in s for kw in _CONSTRAINT_KEYWORDS)]

    user_goal = ""
    for s in reversed(sentences):
        if any(kw in s for kw in _GOAL_KEYWORDS):
            user_goal = s
            break
    if not user_goal:
        user_goal = lines[-1] if lines else ""

    return {
        "project_summary": project_summary,
        "tech_stack": extracted_tech,
        "current_stage": stage or "미정",
        "concerns": concerns,
        "domain": domains,
        "constraints": constraints,
        "user_goal": user_goal,
    }


def _parse_to_structured_llm(
    project_text: str,
    tech_stack: list[str],
    stage: str,
) -> dict[str, Any]:
    """LLM-based structured parser. Swap in when LLM integration is ready."""
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# LangGraph node (A 담당)
# ─────────────────────────────────────────────────────────────

def parse_input(state: GraphState) -> GraphState:
    """Input Parser node.

    Reads: user_input, tech_stack, stage, clarify_answer
    Writes: is_input_sufficient, clarification_question,
            clarification_options, parsed_input
    """
    project_text: str = state.get("user_input", "")  # type: ignore[arg-type]
    tech_stack: list[str] = state.get("tech_stack", [])  # type: ignore[arg-type]
    stage: str = state.get("stage", "")  # type: ignore[arg-type]
    clarify_answer: str | None = state.get("clarify_answer")  # type: ignore[arg-type]

    # clarify_answer가 있으면 병합 후 바로 파싱 (추가 질문 없음)
    if clarify_answer:
        merged = _merge_clarify_answer(project_text, clarify_answer)
        return {
            **state,
            "user_input": merged,
            "is_input_sufficient": True,
            "clarification_question": "",
            "clarification_options": [],
            "parsed_input": _parse_to_structured_rule_based(merged, tech_stack, stage),
        }

    is_sufficient, question, options = _check_sufficiency_rule_based(project_text, tech_stack)

    if not is_sufficient:
        return {
            **state,
            "is_input_sufficient": False,
            "clarification_question": question,
            "clarification_options": options,
            "parsed_input": {},
        }

    return {
        **state,
        "is_input_sufficient": True,
        "clarification_question": "",
        "clarification_options": [],
        "parsed_input": _parse_to_structured_rule_based(project_text, tech_stack, stage),
    }
