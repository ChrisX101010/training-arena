"""Bootcamp v3 - Self-evolution with tree-based exploration.

Inspired by TREX (Automating LLM Fine-tuning via Agent-Driven Tree-based
Exploration) and MiniMax M2.7's recursive self-optimization.

Key improvements over v2:
  1. TREE SEARCH — each round is a node; if a round makes things worse,
     we backtrack and try a different strategy (different domain, more data)
  2. COMPOUNDING MEMORY — results from previous runs persist and inform
     future strategy (which domains improved, which stalled)
  3. AXIOM WIKI — pull pre-compiled knowledge before each round
  4. MULTI-SOURCE DATA — axiom pages + corrections + wikipedia + teacher synthesis
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from core.model_hub import ModelHub
from core.metrics_db import MetricsDatabase
from core.evaluation_rubric import EvaluationRubric
from core.trainer import DistillationTrainer
from core.llm_wiki import LLMWiki
from core.rehearsal_gym import RehearsalGym
from core.larql_integration import get_larql


class Bootcamp:
    """Self-evolution with tree-based exploration and compounding memory."""

    DOMAIN_PROBES = {
        "knowledge": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "Name three primary colors.",
        ],
        "explanation": [
            "Explain the concept of gravity in simple terms.",
            "What is the boiling point of water in Celsius?",
        ],
        "math": [
            "What is 2 + 2?",
            "Solve: If x + 3 = 7, what is x?",
        ],
        "creative": [
            "Write a short poem about a cat.",
            "Write a haiku about autumn.",
        ],
        "coding": [
            "Write a Python function to reverse a string.",
            "What is recursion?",
        ],
    }

    def __init__(self, config_path: str = "config/models.yaml"):
        self.hub = ModelHub(config_path)
        self.db = MetricsDatabase()
        self.rubric = EvaluationRubric(judge_model_name="phi3:mini")
        self.trainer = DistillationTrainer(config_path)
        self.wiki = LLMWiki()
        self.gym = RehearsalGym(self.hub)
        self.larql = get_larql()

        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def run(self, student: str, teacher: str,
            max_rounds: int = 5, target_delta: float = 0.15,
            run_rehearsal: bool = True) -> Dict:

        print("\n" + "=" * 60)
        print("🏋️  BOOTCAMP v3 — Self-Evolution (TREX + Compounding)")
        print(f"   Student: {student}")
        print(f"   Teacher: {teacher}")
        print("=" * 60)

        # ── Phase 0: Pull ALL knowledge sources ──
        print("\n📥 Pulling knowledge from all sources…")
        pull_msg = self.wiki.pull()
        print(f"   Git: {pull_msg}")

        # Axiom Wiki (highest quality — pre-compiled)
        if self.wiki.axiom_available:
            for domain in self.DOMAIN_PROBES:
                count = self.wiki.ingest_from_axiom(domain, domain, max_pages=5)
            print(f"   Axiom: ingested pages")

        # Fetch Wikipedia for domains that are thin
        stats = self.wiki.get_wiki_stats()
        for domain, count in stats["domains"].items():
            if count < 3 and domain in ["science", "math", "coding"]:
                topics = {"science": ["gravity", "photosynthesis", "DNA"],
                          "math": ["Pythagorean theorem", "algebra", "calculus"],
                          "coding": ["recursion", "algorithm", "data structure"]}
                fetched = self.wiki.fetch_batch(topics.get(domain, []), domain)
                if fetched:
                    print(f"   Wikipedia: +{fetched} for {domain}")

        # Build enriched probes
        probes = dict(self.DOMAIN_PROBES)
        for domain in probes:
            wiki_prompts = self.wiki.generate_training_prompts(domain, 5)
            corrections = self.wiki.get_training_data_from_corrections(domain)
            extras = self._extract_questions(wiki_prompts + corrections)
            if extras:
                probes[domain] = probes[domain] + extras
                print(f"   +{len(extras)} enriched prompts for {domain}")

        # ── Evolution loop (tree search) ──
        history = []
        tree = []  # TREX-inspired: track each node for backtracking
        baseline = None
        best_score = None
        best_round = None

        for rnd in range(1, max_rounds + 1):
            print(f"\n── Round {rnd}/{max_rounds} ──")

            # 1. EVALUATE
            print("  📊 Evaluating…")
            scores = self._evaluate(student, probes)
            overall = sum(scores.values()) / max(len(scores), 1)

            if baseline is None:
                baseline = overall
                best_score = overall
                print(f"  📏 Baseline: {overall:.3f}")
            else:
                delta = overall - baseline
                print(f"  📏 Score: {overall:.3f}  (Δ {delta:+.3f} from baseline)")

                # TREX: Check if this round improved or regressed
                if overall < best_score - 0.02:
                    print(f"  ⚠️ Regression detected! Backtracking strategy…")
                    # Don't train on the same weakest domain — try second weakest
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
                    weakest = sorted_scores[1][0] if len(sorted_scores) > 1 else sorted_scores[0][0]
                    print(f"  🔄 Switching to: {weakest}")
                else:
                    weakest = min(scores, key=scores.get)
                    best_score = max(best_score, overall)
                    best_round = rnd

                if delta >= target_delta:
                    print(f"  ✅ Target reached!")
                    history.append(self._node(rnd, scores, overall, None, 0))
                    break

            if baseline == overall:
                weakest = min(scores, key=scores.get)

            print(f"  🔍 Targeting: {weakest} ({scores[weakest]:.3f})")

            # 2. SYNTHESIZE from ALL sources
            print(f"  🧪 Synthesizing multi-source training data…")
            training_texts = self._synthesize_multi_source(teacher, weakest, probes[weakest], 15)
            print(f"     {len(training_texts)} total examples")

            # 3. DISTILL
            output_path = f"./results/bootcamp/{student.replace('/', '_')}/round_{rnd}"
            print("  🔥 Distilling…")
            self.trainer.distill(teacher, student, training_texts, output_path, num_epochs=2)

            # 3b. LARQL VERIFICATION (if available)
            if self.larql.available:
                vindex = self.larql.extract_index(output_path + "/final",
                                                   f"bootcamp_r{rnd}.vindex")
                if vindex:
                    concepts = list(probes.get(weakest, probes["knowledge"]))[:3]
                    concept_words = [c.split()[-1].rstrip("?.")
                                    for c in concepts if len(c.split()) > 2]
                    if concept_words:
                        verifications = self.larql.verify_knowledge(vindex, concept_words)
                        verified = sum(1 for v in verifications if v.found)
                        print(f"  🔬 Weight verification: {verified}/{len(concept_words)} concepts found")

            # 4. RECORD
            self.db.record_training_run(teacher, student, "bootcamp", rnd,
                                        len(training_texts), output_path)
            self.db.record_evolution_step(student, rnd, weakest,
                                          overall - baseline, len(training_texts))

            node = self._node(rnd, scores, overall, weakest, len(training_texts))
            history.append(node)
            tree.append({**node, "improved": overall >= (best_score - 0.02),
                        "strategy": f"target_{weakest}"})

        final = history[-1]["overall"] if history else baseline

        # 5. REHEARSAL
        rehearsal_result = None
        if run_rehearsal:
            print("\n" + "=" * 60)
            print("🧪 REHEARSAL — Student vs Teacher")
            print("=" * 60)
            rehearsal_prompts = []
            for dp in probes.values():
                rehearsal_prompts.extend(dp[:2])
            rehearsal_result = self.gym.benchmark_and_save(
                student, teacher, rehearsal_prompts[:10])

        result = {
            "student": student, "teacher": teacher,
            "baseline": round(baseline, 4),
            "final": round(final, 4),
            "improvement": round(final - baseline, 4),
            "best_score": round(best_score or baseline, 4),
            "best_round": best_round,
            "rounds": len(history),
            "history": history,
            "tree": tree,
            "rehearsal": rehearsal_result,
            "ready_for_arena": rehearsal_result["ready_for_arena"] if rehearsal_result else None,
            "knowledge_sources": {
                "axiom_pages": self.wiki.get_wiki_stats().get("axiom_pages", 0),
                "corrections": self.wiki.get_wiki_stats().get("corrections", 0),
                "wiki_articles": self.wiki.get_wiki_stats().get("total_articles", 0),
            },
            "timestamp": datetime.now().isoformat(),
        }

        out_path = Path(f"./results/bootcamp/{student.replace('/', '_')}")
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "bootcamp_report.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n🏋️  Bootcamp complete: {baseline:.3f} → {final:.3f}  "
              f"(Δ {final - baseline:+.3f})")
        if best_round:
            print(f"   Best score: {best_score:.3f} at round {best_round}")
        return result

    # ------------------------------------------------------------------
    def _evaluate(self, model: str, probes: Dict[str, List[str]]) -> Dict[str, float]:
        out = {}
        for domain, prompts in probes.items():
            domain_scores = []
            for prompt in prompts[:4]:
                try:
                    resp = self.hub.generate_response(model, prompt, max_tokens=100)
                    rubric = self.rubric.evaluate_response(self.hub, prompt, resp, domain)
                    domain_scores.append(self.rubric.calculate_overall(rubric, domain))
                except Exception:
                    domain_scores.append(0.5)
            out[domain] = sum(domain_scores) / max(len(domain_scores), 1)
        return out

    def _synthesize_multi_source(self, teacher: str, domain: str,
                                  existing: List[str], count: int = 15) -> List[str]:
        """Combine all knowledge sources into training data."""
        texts = []

        # Source 1: Axiom Wiki compiled pages (highest quality)
        axiom_prompts = self.wiki.generate_training_prompts(domain, 5)
        texts.extend(axiom_prompts)

        # Source 2: Human corrections (gold standard)
        correction_texts = self.wiki.get_training_data_from_corrections(domain)
        texts.extend(correction_texts)

        # Source 3: Teacher-generated answers to probe questions
        meta = (
            f"Generate {count - len(texts)} diverse questions about '{domain}' "
            f"testing deep understanding. One per line."
        )
        try:
            raw = self.hub.generate_response(teacher, meta, max_tokens=400)
            questions = [l.strip().lstrip("0123456789.-) ")
                        for l in raw.split("\n") if l.strip()][:count - len(texts)]
        except Exception:
            questions = existing[:3]

        for q in questions:
            try:
                a = self.hub.generate_response(teacher, q, max_tokens=100)
                texts.append(f"Question: {q}\nAnswer: {a}")
            except Exception:
                pass

        return texts[:count] or [f"Question: {q}\nAnswer: (no response)" for q in existing[:3]]

    @staticmethod
    def _extract_questions(training_texts: List[str]) -> List[str]:
        out = []
        for t in training_texts:
            if t.startswith("Question:"):
                out.append(t.split("\n", 1)[0].replace("Question:", "").strip())
            else:
                out.append(t.split("\n")[0])
        return out

    @staticmethod
    def _node(rnd, scores, overall, gap, synth_count):
        return {"round": rnd, "scores": {k: round(v, 4) for k, v in scores.items()},
                "overall": round(overall, 4), "gap": gap, "synthetic_count": synth_count}
