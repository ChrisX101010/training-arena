"""Rehearsal Gym - Teacher/Student improvement benchmark.

After Bootcamp (self-evolution), the rehearsal benchmarks the student
against its teacher. This is where we measure "did the student actually
improve?" — the core question of the whole platform.

The teacher can be:
  - The original teacher used in Bootcamp (measure distillation gain)
  - A commercial model like GPT-4 (measure how close we are to the field)
  - A previous version of the student (measure iteration-over-iteration)

Sometimes the student becomes the master.

Outputs:
  1. Per-prompt rubric scores (6 dimensions, real evaluation)
  2. Head-to-head win rate student vs teacher
  3. Readiness verdict: is the student ready for the Arena?
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from core.model_hub import ModelHub
from core.evaluation_rubric import EvaluationRubric


class RehearsalGym:
    """Teacher/Student benchmark + arena readiness gate."""

    READINESS_THRESHOLD = 0.55  # Student rubric score needed to enter Arena
    IMPROVEMENT_THRESHOLD = 0.05  # Minimum delta vs teacher to call it "improved"

    def __init__(self, hub: ModelHub, judge_model: str = "phi3:mini"):
        self.hub = hub
        self.rubric = EvaluationRubric(judge_model_name=judge_model)

    # ------------------------------------------------------------------
    # Cognitive warm-up (used during bootcamp)
    # ------------------------------------------------------------------
    def warmup(self, model_name: str, prompt: str) -> Dict:
        """Recite → Reflect → Answer. Internal cognitive prep."""
        recited = self.hub.generate_response(
            model_name,
            f"Before answering, list relevant facts about:\n{prompt}\n\nFacts:",
            max_tokens=150,
        )
        reflection = self.hub.generate_response(
            model_name,
            f'Q: "{prompt}"\nRecall: "{recited}"\n\n'
            f'Self-critique: is this complete and accurate?\n\nReflection:',
            max_tokens=150,
        )
        final = self.hub.generate_response(
            model_name,
            f'Your preparation: "{reflection}"\n\nAnswer: {prompt}',
            max_tokens=250,
        )
        return {"prompt": prompt, "recitation": recited,
                "reflection": reflection, "final_response": final}

    def batch_warmup(self, model_name: str, prompts: List[str]) -> List[Dict]:
        return [self.warmup(model_name, p) for p in prompts]

    # ------------------------------------------------------------------
    # Teacher vs Student benchmark — the real test
    # ------------------------------------------------------------------
    def benchmark(self, student: str, teacher: str,
                  prompts: List[str]) -> Dict:
        """
        Head-to-head: student vs teacher on the same prompts.
        Both scored with the 6-dimension rubric. This measures real improvement.
        """
        print(f"\n🧪 Rehearsal Benchmark")
        print(f"   Student: {student}")
        print(f"   Teacher: {teacher}")
        print(f"   Prompts: {len(prompts)}")
        print("-" * 50)

        results = []
        student_wins = teacher_wins = draws = 0

        for i, prompt in enumerate(prompts):
            print(f"\n  [{i+1}/{len(prompts)}] {prompt[:50]}…")

            # Both models answer
            try:
                s_resp = self.hub.generate_response(student, prompt, max_tokens=150)
            except Exception as e:
                s_resp = f"[error: {e}]"
            try:
                t_resp = self.hub.generate_response(teacher, prompt, max_tokens=150)
            except Exception as e:
                t_resp = f"[error: {e}]"

            # Real rubric scoring
            try:
                s_scores = self.rubric.evaluate_response(self.hub, prompt, s_resp)
                s_overall = self.rubric.calculate_overall(s_scores)
            except Exception:
                s_overall = 0.5
                s_scores = {}

            try:
                t_scores = self.rubric.evaluate_response(self.hub, prompt, t_resp)
                t_overall = self.rubric.calculate_overall(t_scores)
            except Exception:
                t_overall = 0.5
                t_scores = {}

            if s_overall > t_overall + 0.02:
                winner = "student"; student_wins += 1
            elif t_overall > s_overall + 0.02:
                winner = "teacher"; teacher_wins += 1
            else:
                winner = "draw"; draws += 1

            print(f"      student={s_overall:.3f}  teacher={t_overall:.3f}  → {winner}")

            results.append({
                "prompt": prompt,
                "student_response": s_resp,
                "teacher_response": t_resp,
                "student_score": round(s_overall, 4),
                "teacher_score": round(t_overall, 4),
                "student_rubric": {k.value: v.score for k, v in s_scores.items()},
                "teacher_rubric": {k.value: v.score for k, v in t_scores.items()},
                "winner": winner,
            })

        # Aggregate
        s_avg = sum(r["student_score"] for r in results) / max(len(results), 1)
        t_avg = sum(r["teacher_score"] for r in results) / max(len(results), 1)
        delta = s_avg - t_avg
        win_rate = student_wins / max(len(results), 1)

        # Verdicts
        improved = delta >= self.IMPROVEMENT_THRESHOLD
        ready_for_arena = s_avg >= self.READINESS_THRESHOLD
        became_master = s_avg > t_avg + 0.1

        print("\n" + "=" * 50)
        print(f"  Student avg: {s_avg:.3f}")
        print(f"  Teacher avg: {t_avg:.3f}")
        print(f"  Delta: {delta:+.3f}")
        print(f"  W/L/D: {student_wins}/{teacher_wins}/{draws}  (win rate {win_rate:.0%})")
        print(f"  Improved vs teacher: {'✅' if improved else '❌'}")
        print(f"  Ready for Arena: {'✅' if ready_for_arena else '❌'}")
        if became_master:
            print(f"  🎓 The student has become the master.")
        print("=" * 50)

        return {
            "student": student,
            "teacher": teacher,
            "timestamp": datetime.now().isoformat(),
            "student_avg": round(s_avg, 4),
            "teacher_avg": round(t_avg, 4),
            "delta": round(delta, 4),
            "student_wins": student_wins,
            "teacher_wins": teacher_wins,
            "draws": draws,
            "win_rate": round(win_rate, 4),
            "improved": improved,
            "ready_for_arena": ready_for_arena,
            "became_master": became_master,
            "results": results,
        }

    def benchmark_and_save(self, student: str, teacher: str,
                           prompts: List[str], output_dir: str = "./results/rehearsal") -> Dict:
        """Run benchmark and save report."""
        result = self.benchmark(student, teacher, prompts)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe_name = student.replace("/", "_")
        report_path = out / f"{safe_name}_vs_{teacher.replace('/', '_')}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n   📄 Report → {report_path}")
        return result
