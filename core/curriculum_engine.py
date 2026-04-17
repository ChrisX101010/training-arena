"""Curriculum Engine - Full pipeline orchestration.

  Bootcamp → self-evolve the student (with rehearsal gate at the end)
  Academy  → train domain specialists (small, quantised, focused)
  Arena    → global leaderboard (all models equal, commercial baselines included)

The pipeline:
  1. Wiki pulls fresh data
  2. Bootcamp self-evolves the student
  3. Rehearsal benchmarks student vs teacher — decides if ready
  4. Academy can train specialists using the evolved student as teacher
  5. Arena ranks everything
"""

import json
from pathlib import Path
from typing import List, Dict

from core.bootcamp import Bootcamp
from core.academy import Academy
from core.arena import Arena


class CurriculumEngine:
    def __init__(self, config_path: str = "config/models.yaml"):
        self.bootcamp = Bootcamp(config_path)
        self.academy = Academy(config_path)
        self.arena = Arena(config_path)
        self.config_path = config_path

    def run_bootcamp(self, student: str = None, teacher: str = None,
                     rounds: int = 5, run_rehearsal: bool = True) -> Dict:
        import yaml
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        student = student or cfg["training"].get("student_model", "distilgpt2")
        teacher = teacher or "Qwen/Qwen2.5-0.5B"
        return self.bootcamp.run(student, teacher, max_rounds=rounds,
                                 run_rehearsal=run_rehearsal)

    def run_academy(self, domains: List[str] = None,
                    teacher: str = None, student: str = None) -> Dict:
        return self.academy.run_all_tracks(domains, teacher, student)

    def run_arena(self, models: List[str] = None,
                  include_baselines: bool = True) -> List[Dict]:
        return self.arena.run_tournament(models, include_commercial_baselines=include_baselines)

    def run_full(self, student: str = None, teacher: str = None,
                 bootcamp_rounds: int = 3, academy_domains: List[str] = None) -> Dict:
        """Full pipeline: Bootcamp → Academy → Arena."""
        out = {}

        # 1. Self-evolve
        out["bootcamp"] = self.run_bootcamp(student, teacher, bootcamp_rounds)

        # 2. Train specialists (uses wiki data)
        out["academy"] = self.run_academy(academy_domains, teacher, student)

        # 3. Global leaderboard
        out["arena"] = self.run_arena()

        Path("./results").mkdir(exist_ok=True)
        with open("./results/curriculum_report.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

        print(f"\n🏆 Full pipeline complete → ./results/curriculum_report.json")
        return out
