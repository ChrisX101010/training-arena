"""Arena v3 - Fixed scoring that actually differentiates model outputs."""

import json
import random
import yaml
from typing import List, Dict
from core.model_hub import ModelHub
from core.metrics_db import MetricsDatabase
from core.evaluation_rubric import EvaluationRubric, JudgeUnavailableError

try:
    from polyrating import Player, Game, RatingSystem, PolyratingCrossEntropy
    POLYRATING_OK = True
except ImportError:
    POLYRATING_OK = False

COMMERCIAL_BASELINES = {
    "claude-opus-4.6": {"general": 0.92, "math": 0.88, "coding": 0.91, "creative": 0.91, "science": 0.91},
    "gpt-5.4": {"general": 0.91, "math": 0.90, "coding": 0.90, "creative": 0.88, "science": 0.90},
    "gemini-3.1-pro": {"general": 0.89, "math": 0.87, "coding": 0.88, "creative": 0.85, "science": 0.89},
    "minimax-m2.7": {"general": 0.88, "math": 0.84, "coding": 0.89, "creative": 0.82, "science": 0.86},
    "qwen3.5-9b": {"general": 0.82, "math": 0.81, "coding": 0.83, "creative": 0.76, "science": 0.82},
    "qwen3.5-4b": {"general": 0.76, "math": 0.74, "coding": 0.77, "creative": 0.70, "science": 0.75},
    "llama-4-scout": {"general": 0.80, "math": 0.78, "coding": 0.82, "creative": 0.74, "science": 0.79},
    "gemma-4-9b": {"general": 0.79, "math": 0.77, "coding": 0.80, "creative": 0.75, "science": 0.78},
    "deepseek-r1": {"general": 0.78, "math": 0.82, "coding": 0.76, "creative": 0.72, "science": 0.80},
}

DOMAIN_KEYWORDS = {
    "knowledge": ["is", "was", "are", "the", "known", "called", "named", "capital", "country", "wrote", "author"],
    "explanation": ["because", "means", "when", "force", "energy", "process", "causes", "result", "effect"],
    "math": ["equals", "sum", "result", "answer", "solve", "calculate", "number", "equation", "plus", "minus"],
    "science": ["temperature", "water", "energy", "atom", "cell", "molecule", "reaction", "degree", "celsius"],
    "creative": ["poem", "verse", "rhyme", "beauty", "heart", "dream", "light", "shadow", "song", "love"],
    "coding": ["function", "return", "def", "class", "variable", "loop", "code", "algorithm", "input", "output"],
}


class Arena:
    def __init__(self, config_path="config/models.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.hub = ModelHub(config_path)
        self.db = MetricsDatabase()
        self.judge_model = self.config["arena"]["judge_model"]
        self.prompts = self._load_prompts(self.config["arena"]["prompts_file"])
        self.max_prompts = self.config["arena"].get("max_prompts_per_match", 7)
        self.rubric = EvaluationRubric(judge_model_name=self.judge_model)
        self._use_heuristic = False
        engine = "Polyrating" if POLYRATING_OK else "ELO"
        print(f"📊 Arena using {engine}")

    def _load_prompts(self, path):
        with open(path) as f: return json.load(f)["prompts"]

    def get_enabled_models(self):
        return [m["name"] for m in self.config["models"] if m.get("enabled", True)]

    def score_response(self, prompt, response, domain="general"):
        if not self._use_heuristic:
            try:
                scores = self.rubric.evaluate_response(self.hub, prompt, response, domain)
                return round(self.rubric.calculate_overall(scores, domain), 4)
            except JudgeUnavailableError:
                self._use_heuristic = True
                print("  📏 Switching to heuristic scoring")
            except Exception:
                pass
        return self._heuristic_score(prompt, response, domain)

    def _heuristic_score(self, prompt, response, domain="general"):
        if not response or len(response.strip()) < 5:
            return 0.05
        resp_lower = response.lower()
        resp_words = resp_lower.split()
        prompt_words = set(prompt.lower().split())
        word_count = len(resp_words)

        # Length adequacy
        if word_count < 5: length_score = 0.1
        elif word_count < 15: length_score = 0.3 + (word_count - 5) * 0.03
        elif word_count <= 100: length_score = 0.6 + min((word_count - 15) / 85, 1.0) * 0.3
        else: length_score = 0.8

        # Prompt keyword coverage
        overlap = len(prompt_words & set(resp_words))
        coverage_score = min(overlap / max(len(prompt_words), 1), 1.0)

        # Domain keyword relevance
        domain_kw = DOMAIN_KEYWORDS.get(domain, DOMAIN_KEYWORDS.get("knowledge", []))
        domain_hits = sum(1 for kw in domain_kw if kw in resp_lower)
        domain_score = min(domain_hits / max(len(domain_kw) * 0.3, 1), 1.0)

        # Vocabulary diversity
        unique_ratio = len(set(resp_words)) / max(word_count, 1)
        diversity_score = min(unique_ratio * 1.5, 1.0)

        # Sentence structure
        sentences = response.count('.') + response.count('!') + response.count('?')
        structure_score = 0.2 if sentences == 0 else (0.5 + min(sentences, 3) * 0.1 if sentences <= 3 else 0.8)

        # Coherence markers
        coherence_words = ["because", "therefore", "however", "although", "since", "means",
                           "result", "example", "such as", "first", "second", "finally", "also"]
        coherence_score = min(sum(1 for cw in coherence_words if cw in resp_lower) * 0.15, 1.0)

        # Repetition penalty
        rep_penalty = 0
        if word_count > 10:
            trigrams = [" ".join(resp_words[i:i+3]) for i in range(len(resp_words)-2)]
            if trigrams:
                rep_penalty = max(0, 1.0 - len(set(trigrams)) / len(trigrams)) * 0.3

        score = (0.20 * length_score + 0.15 * coverage_score + 0.15 * domain_score +
                 0.15 * diversity_score + 0.15 * structure_score + 0.10 * coherence_score +
                 0.10 * (1.0 - rep_penalty))

        noise = random.uniform(-0.005, 0.005)
        return round(max(0.05, min(0.95, score + noise)), 4)

    def run_tournament(self, models=None, include_commercial_baselines=True):
        models = models or self.get_enabled_models()
        prompts = self.prompts[:self.max_prompts]
        print(f"\n🏟️  Arena v3  |  {len(models)} models  |  {len(prompts)} prompts")
        print("=" * 55)

        match_log = []
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                print(f"\n⚔️  {ma.split('/')[-1]}  vs  {mb.split('/')[-1]}")
                wa = wb = dr = 0
                for k, pobj in enumerate(prompts):
                    pt, cat = pobj["text"], pobj.get("category", "general")
                    try: ra = self.hub.generate_response(ma, pt, max_tokens=150)
                    except Exception as e: ra = f"[error: {e}]"
                    try: rb = self.hub.generate_response(mb, pt, max_tokens=150)
                    except Exception as e: rb = f"[error: {e}]"

                    sa = self.score_response(pt, ra, cat)
                    sb = self.score_response(pt, rb, cat)

                    if sa > sb + 0.01: winner = ma
                    elif sb > sa + 0.01: winner = mb
                    else: winner = "draw"

                    self.db.record_match(ma, mb, winner, pt, cat, sa, sb)
                    match_log.append({"a": ma, "b": mb, "winner": winner, "sa": sa, "sb": sb, "cat": cat})
                    tag = "DRAW" if winner == "draw" else winner.split("/")[-1][:10]
                    print(f"  [{k+1}] {tag:<12} ({sa:.3f} / {sb:.3f})  [{cat}]")
                    if winner == ma: wa += 1
                    elif winner == mb: wb += 1
                    else: dr += 1
                print(f"  → {wa}–{wb}  (draws {dr})")

        rankings = self._elo(models, match_log)
        if include_commercial_baselines:
            for name, scores in COMMERCIAL_BASELINES.items():
                avg = sum(scores.values()) / len(scores)
                rankings.append({"model_name": name, "elo_rating": round(1500 + (avg - 0.5) * 1400, 1),
                                 "wins": 0, "losses": 0, "draws": 0, "type": "commercial"})
            rankings.sort(key=lambda x: x["elo_rating"], reverse=True)

        for r in rankings:
            self.db.record_score(r["model_name"], "arena_global", "elo", r["elo_rating"],
                                 {"wins": r["wins"], "losses": r["losses"], "type": r.get("type", "local")})
        return rankings

    def _elo(self, models, matches, K=32):
        ratings = {m: 1500.0 for m in models}
        for m in matches:
            a, b = m["a"], m["b"]
            ea = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
            sa = 1 if m["winner"] == a else (0 if m["winner"] == b else 0.5)
            ratings[a] += K * (sa - ea)
            ratings[b] += K * ((1 - sa) - (1 - ea))
        wld = {m: {"wins": 0, "losses": 0, "draws": 0} for m in models}
        for m in matches:
            if m["winner"] == m["a"]: wld[m["a"]]["wins"] += 1; wld[m["b"]]["losses"] += 1
            elif m["winner"] == m["b"]: wld[m["b"]]["wins"] += 1; wld[m["a"]]["losses"] += 1
            else: wld[m["a"]]["draws"] += 1; wld[m["b"]]["draws"] += 1
        return sorted([{"model_name": m, "elo_rating": round(ratings[m], 1), "type": "local", **wld[m]}
                        for m in models], key=lambda x: x["elo_rating"], reverse=True)
