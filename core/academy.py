"""Academy - Train domain-specialist models.

Each Academy track produces a focused, smaller model optimised for one
use-case (math, legal, medical, coding, etc.). The workflow:

  1. PULL wiki data for the domain
  2. SELECT best teacher for the domain (from Bootcamp graduates or config)
  3. DISTILL into a small student (e.g. distilgpt2)
  4. EVALUATE with domain-weighted rubric
  5. EXPORT the specialist model

Academy models are intended to be small enough to quantise and deploy
at the edge — the opposite of Bootcamp's generalist self-evolution.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from core.model_hub import ModelHub
from core.metrics_db import MetricsDatabase
from core.evaluation_rubric import EvaluationRubric
from core.trainer import DistillationTrainer
from core.llm_wiki import LLMWiki


class Academy:
    """Domain-specialist training: small, quantised, production-ready models."""

    # Default domain tracks
    TRACKS = {
        "math": {
            "prompts": [
                "What is 2 + 2? Show your reasoning.",
                "Solve: If x + 3 = 7, what is x?",
                "Explain the Pythagorean theorem.",
                "What is the square root of 144?",
                "Calculate 15% of 200.",
            ],
            "description": "Mathematical reasoning and computation",
        },
        "science": {
            "prompts": [
                "What is the boiling point of water in Celsius?",
                "Explain how photosynthesis works.",
                "What are the three states of matter?",
                "What is DNA?",
                "Explain Newton's first law.",
            ],
            "description": "Scientific knowledge and explanation",
        },
        "coding": {
            "prompts": [
                "Write a Python function to reverse a string.",
                "What is recursion in programming?",
                "Explain the difference between a list and a tuple in Python.",
                "Write a function to check if a number is prime.",
                "What is an API?",
            ],
            "description": "Programming and software development",
        },
        "creative": {
            "prompts": [
                "Write a short poem about the ocean.",
                "Write a haiku about autumn.",
                "Create a metaphor for loneliness.",
                "Write the opening line of a mystery novel.",
                "Describe a sunset without using the word 'red'.",
            ],
            "description": "Creative writing and expression",
        },
    }

    def __init__(self, config_path: str = "config/models.yaml"):
        self.hub = ModelHub(config_path)
        self.db = MetricsDatabase()
        self.rubric = EvaluationRubric()
        self.trainer = DistillationTrainer(config_path)
        self.wiki = LLMWiki()

        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def train_specialist(self, domain: str, teacher: str = None,
                         student: str = None) -> Dict:
        """Train a domain-specialist model."""
        student = student or self.config["training"].get("student_model", "distilgpt2")
        teacher = teacher or self.db.recommend_model(domain) or "Qwen/Qwen2.5-0.5B"

        track = self.TRACKS.get(domain, {
            "prompts": [f"Explain a concept in {domain}."],
            "description": domain,
        })

        print(f"\n🎓 ACADEMY — {domain.upper()} Track")
        print(f"   {track['description']}")
        print(f"   Teacher: {teacher}  →  Student: {student}")
        print("-" * 50)

        # 1. Pull fresh wiki data
        wiki_prompts = self.wiki.generate_training_prompts(domain, 20)
        all_prompts = track["prompts"] + (wiki_prompts or [])
        print(f"   Prompts: {len(track['prompts'])} built-in + {len(wiki_prompts or [])} wiki = {len(all_prompts)}")

        # 2. Generate teacher answers
        print("   📝 Teacher generating answers…")
        training_texts = []
        for p in all_prompts:
            try:
                a = self.hub.generate_response(teacher, p, max_tokens=100)
                training_texts.append(f"Question: {p}\nAnswer: {a}")
            except Exception:
                pass

        # 3. Distill
        output_path = f"./results/academy/{domain}/{student}"
        epochs = self.config["training"].get("num_epochs", 3)
        self.trainer.distill(teacher, student, training_texts, output_path, epochs)

        # 4. Record
        self.db.record_training_run(teacher, student, "academy", 0, len(training_texts), output_path)

        result = {
            "domain": domain, "teacher": teacher, "student": student,
            "prompts_used": len(training_texts), "output_path": output_path,
        }

        # 5. Save report
        report_path = Path(output_path) / "specialist_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"   ✅ {domain} specialist saved → {output_path}")
        return result

    def run_all_tracks(self, domains: List[str] = None, teacher: str = None,
                       student: str = None) -> Dict:
        """Train specialists for multiple domains."""
        domains = domains or list(self.TRACKS.keys())

        print("\n" + "=" * 60)
        print("🎓 ACADEMY — Domain Specialisation")
        print("=" * 60)

        # Pull wiki once
        self.wiki.pull()

        results = {}
        for domain in domains:
            results[domain] = self.train_specialist(domain, teacher, student)

        Path("./results/academy").mkdir(parents=True, exist_ok=True)
        with open("./results/academy/results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✅ Academy complete — {len(results)} specialists trained")
        return results
