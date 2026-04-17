"""Evaluation Rubric - Multi-dimensional scoring (inspired by NVIDIA QCalEval)."""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class QuestionType(Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"
    REASONING_QUALITY = "reasoning_quality"
    SOURCE_ATTRIBUTION = "source_attribution"
    HALLUCINATION_DETECTION = "hallucination_detection"


@dataclass
class RubricScore:
    question_type: str
    score: float
    confidence: float
    reasoning: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    model_name: str
    prompt: str
    response: str
    domain: str
    scores: List[RubricScore]
    overall_score: float
    timestamp: str
    judge_model: str


class EvaluationRubric:
    RUBRIC_PROMPTS = {
        QuestionType.FACTUAL_ACCURACY: (
            "Evaluate factual accuracy (0.0-1.0). Consider: verifiable statements, false claims, numerical accuracy.\n"
            'Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}'
        ),
        QuestionType.COMPLETENESS: (
            "Evaluate completeness (0.0-1.0). Consider: all aspects covered, omissions, depth.\n"
            'Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}'
        ),
        QuestionType.CONCISENESS: (
            "Evaluate conciseness (0.0-1.0). Consider: repetition, verbosity, density.\n"
            'Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}'
        ),
        QuestionType.REASONING_QUALITY: (
            "Evaluate reasoning quality (0.0-1.0). Consider: logical structure, conclusions, fallacies.\n"
            'Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}'
        ),
        QuestionType.SOURCE_ATTRIBUTION: (
            "Evaluate source attribution (0.0-1.0). Consider: references, fact vs inference.\n"
            'Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}'
        ),
        QuestionType.HALLUCINATION_DETECTION: (
            "Evaluate hallucinations (0.0-1.0, higher = fewer). Consider: contradictions, invented details.\n"
            'Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}'
        ),
    }

    DEFAULT_WEIGHTS = {
        QuestionType.FACTUAL_ACCURACY: 0.30,
        QuestionType.COMPLETENESS: 0.20,
        QuestionType.CONCISENESS: 0.10,
        QuestionType.REASONING_QUALITY: 0.20,
        QuestionType.SOURCE_ATTRIBUTION: 0.10,
        QuestionType.HALLUCINATION_DETECTION: 0.10,
    }

    DOMAIN_WEIGHTS = {
        "math": {QuestionType.FACTUAL_ACCURACY: 0.40, QuestionType.REASONING_QUALITY: 0.35,
                 QuestionType.COMPLETENESS: 0.15, QuestionType.CONCISENESS: 0.05,
                 QuestionType.HALLUCINATION_DETECTION: 0.05},
        "science": {QuestionType.FACTUAL_ACCURACY: 0.35, QuestionType.REASONING_QUALITY: 0.25,
                    QuestionType.COMPLETENESS: 0.15, QuestionType.SOURCE_ATTRIBUTION: 0.15,
                    QuestionType.CONCISENESS: 0.05, QuestionType.HALLUCINATION_DETECTION: 0.05},
    }

    def __init__(self, judge_model_name: str = "phi3:mini",
                 enabled_dimensions: List[QuestionType] = None):
        self.judge_model_name = judge_model_name
        self.enabled_dimensions = enabled_dimensions or list(QuestionType)

    def evaluate_response(self, hub, prompt: str, response: str,
                          domain: str = "general") -> Dict[QuestionType, RubricScore]:
        scores = {}
        for qtype in self.enabled_dimensions:
            eval_prompt = (
                f"{self.RUBRIC_PROMPTS[qtype]}\n\n"
                f"Original Prompt: {prompt}\nResponse: {response}\n\nOutput ONLY valid JSON:"
            )
            try:
                raw = hub.generate_response(self.judge_model_name, eval_prompt, max_tokens=150)
                parsed = self._parse_json(raw)
                scores[qtype] = RubricScore(
                    question_type=qtype.value,
                    score=parsed.get("score", 0.5),
                    confidence=parsed.get("confidence", 0.5),
                    reasoning=parsed.get("reasoning", ""),
                )
            except Exception as e:
                scores[qtype] = RubricScore(qtype.value, 0.5, 0.3, f"Error: {e}")
        return scores

    def calculate_overall(self, scores, domain: str = "general") -> float:
        w = self.DOMAIN_WEIGHTS.get(domain, self.DEFAULT_WEIGHTS)
        total = sum(scores[q].score * w.get(q, 0.1) for q in scores if q in w)
        denom = sum(w.get(q, 0.1) for q in scores if q in w)
        return total / denom if denom > 0 else 0.5

    @staticmethod
    def _parse_json(text: str) -> Dict:
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"score": 0.5, "confidence": 0.5, "reasoning": "Parse failed"}
