"""Evaluation Rubric - 6-dimension scoring.

If the LLM judge is unavailable, raises JudgeUnavailableError
so the caller falls back to heuristic scoring.
"""

import json
import re
from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum


class JudgeUnavailableError(Exception):
    pass


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


class EvaluationRubric:
    RUBRIC_PROMPTS = {
        QuestionType.FACTUAL_ACCURACY: 'Evaluate factual accuracy (0.0-1.0). Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}',
        QuestionType.COMPLETENESS: 'Evaluate completeness (0.0-1.0). Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}',
        QuestionType.CONCISENESS: 'Evaluate conciseness (0.0-1.0). Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}',
        QuestionType.REASONING_QUALITY: 'Evaluate reasoning quality (0.0-1.0). Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}',
        QuestionType.SOURCE_ATTRIBUTION: 'Evaluate source attribution (0.0-1.0). Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}',
        QuestionType.HALLUCINATION_DETECTION: 'Evaluate hallucinations (0.0-1.0, higher = fewer). Return JSON: {"score": <float>, "confidence": <float>, "reasoning": "<brief>"}',
    }

    DEFAULT_WEIGHTS = {
        QuestionType.FACTUAL_ACCURACY: 0.30, QuestionType.COMPLETENESS: 0.20,
        QuestionType.CONCISENESS: 0.10, QuestionType.REASONING_QUALITY: 0.20,
        QuestionType.SOURCE_ATTRIBUTION: 0.10, QuestionType.HALLUCINATION_DETECTION: 0.10,
    }

    DOMAIN_WEIGHTS = {
        "math": {QuestionType.FACTUAL_ACCURACY: 0.40, QuestionType.REASONING_QUALITY: 0.35,
                 QuestionType.COMPLETENESS: 0.15, QuestionType.CONCISENESS: 0.05,
                 QuestionType.HALLUCINATION_DETECTION: 0.05},
        "science": {QuestionType.FACTUAL_ACCURACY: 0.35, QuestionType.REASONING_QUALITY: 0.25,
                    QuestionType.COMPLETENESS: 0.15, QuestionType.SOURCE_ATTRIBUTION: 0.15,
                    QuestionType.CONCISENESS: 0.05, QuestionType.HALLUCINATION_DETECTION: 0.05},
    }

    def __init__(self, judge_model_name="phi3:mini", enabled_dimensions=None):
        self.judge_model_name = judge_model_name
        self.enabled_dimensions = enabled_dimensions or list(QuestionType)
        self._judge_tested = False
        self._judge_available = False

    def evaluate_response(self, hub, prompt, response, domain="general"):
        if not self._judge_tested:
            self._judge_tested = True
            try:
                hub.generate_response(self.judge_model_name, "test", max_tokens=5)
                self._judge_available = True
            except Exception:
                self._judge_available = False
        if not self._judge_available:
            raise JudgeUnavailableError(f"Judge {self.judge_model_name} not available")

        scores = {}
        for qtype in self.enabled_dimensions:
            eval_prompt = f"{self.RUBRIC_PROMPTS[qtype]}\n\nPrompt: {prompt}\nResponse: {response}\n\nJSON:"
            try:
                raw = hub.generate_response(self.judge_model_name, eval_prompt, max_tokens=150)
                parsed = self._parse_json(raw)
                scores[qtype] = RubricScore(qtype.value, max(0.0, min(1.0, parsed.get("score", 0.5))),
                                            parsed.get("confidence", 0.5), parsed.get("reasoning", ""))
            except Exception as e:
                scores[qtype] = RubricScore(qtype.value, 0.5, 0.3, f"Error: {e}")
        return scores

    def calculate_overall(self, scores, domain="general"):
        w = self.DOMAIN_WEIGHTS.get(domain, self.DEFAULT_WEIGHTS)
        total = sum(scores[q].score * w.get(q, 0.1) for q in scores if q in w)
        denom = sum(w.get(q, 0.1) for q in scores if q in w)
        return total / denom if denom > 0 else 0.5

    @staticmethod
    def _parse_json(text):
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
        return {"score": 0.5, "confidence": 0.5, "reasoning": "Parse failed"}
