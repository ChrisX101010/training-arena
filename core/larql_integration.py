"""LARQL Integration — Weight-level knowledge verification.

LARQL (github.com/chrishayuk/larql) decompiles transformer models into
a queryable vindex (vector index), then provides LQL to browse, edit,
and recompile the model's knowledge at the weight level.

This module integrates LARQL into Training Arena's bootcamp loop:
  1. After distillation: extract-index the student model into a vindex
  2. DESCRIBE target concepts to verify knowledge landed in weights
  3. Compare before/after vindexes to measure actual knowledge gain
  4. Optionally INSERT corrections directly into weights (no retraining)

This is fundamentally better than text-based evaluation — it's the
difference between giving a student a quiz vs scanning their brain.

Requires: cargo install larql (Rust toolchain)
When LARQL is not available, all methods gracefully return None.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class KnowledgeEdge:
    """A single fact stored in the model's weights."""
    entity: str
    relation: str
    target: str
    strength: float
    layer: int
    source: str  # "probe", "walk", "insert"


@dataclass
class KnowledgeVerification:
    """Result of verifying whether target knowledge exists in model weights."""
    concept: str
    found: bool
    edges: List[KnowledgeEdge]
    confidence: float  # 0-1, based on edge strength


class LARQLClient:
    """Client for LARQL weight-level knowledge operations.

    All methods are no-ops when LARQL is not installed.
    This allows the codebase to reference LARQL everywhere
    without crashing on systems that don't have Rust.
    """

    def __init__(self, vindex_dir: str = "./vindexes"):
        self.vindex_dir = Path(vindex_dir)
        self.vindex_dir.mkdir(parents=True, exist_ok=True)
        self.available = self._check_available()
        if self.available:
            print("🔬 LARQL available — weight-level knowledge verification enabled")
        else:
            print("ℹ️  LARQL not installed — using text-based evaluation")
            print("   Install: cargo install larql (requires Rust)")

    def _check_available(self) -> bool:
        """Check if larql CLI is installed."""
        return shutil.which("larql") is not None

    def _run(self, args: List[str], timeout: int = 120) -> Optional[str]:
        """Run a larql CLI command."""
        if not self.available:
            return None
        try:
            result = subprocess.run(
                ["larql"] + args,
                capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"    ⚠️ LARQL error: {result.stderr.strip()[:200]}")
                return None
        except subprocess.TimeoutExpired:
            print("    ⚠️ LARQL timed out")
            return None
        except Exception as e:
            print(f"    ⚠️ LARQL failed: {e}")
            return None

    # ==================================================================
    # EXTRACT — Decompile model into queryable vindex
    # ==================================================================
    def extract_index(self, model_path: str, vindex_name: str = None,
                      level: str = "browse", use_f16: bool = True) -> Optional[str]:
        """Extract a model into a vindex for querying.

        Args:
            model_path: HuggingFace model name or local path
            vindex_name: Output vindex name (default: derived from model)
            level: "browse" (3GB, fast) or "inference" (6GB, full)
            use_f16: Half precision (halves size)

        Returns:
            Path to vindex directory, or None if LARQL unavailable
        """
        if not self.available:
            return None

        if vindex_name is None:
            vindex_name = model_path.replace("/", "_").replace(".", "_") + ".vindex"

        output_path = str(self.vindex_dir / vindex_name)

        args = ["extract-index", model_path, "-o", output_path, "--level", level]
        if use_f16:
            args.append("--f16")

        print(f"  🔬 Extracting vindex: {model_path} → {vindex_name}")
        result = self._run(args, timeout=600)  # Can take a while
        if result is not None:
            print(f"     ✅ Vindex ready: {output_path}")
            return output_path
        return None

    # ==================================================================
    # DESCRIBE — What does the model know about a concept?
    # ==================================================================
    def describe(self, vindex_path: str, concept: str) -> Optional[List[KnowledgeEdge]]:
        """Query what the model knows about a concept at the weight level.

        Example output from LARQL:
          France Edges (L14-27):
            capital → Paris       1436.9  L27 (probe)
            language → French       35.2  L24 (probe)
            continent → Europe      14.4  L25 (probe)
        """
        if not self.available:
            return None

        lql = f'USE "{vindex_path}"; DESCRIBE "{concept}";'
        result = self._run(["lql", lql])
        if result is None:
            return None

        edges = []
        for line in result.split("\n"):
            line = line.strip()
            if "→" in line:
                try:
                    parts = line.split("→")
                    relation = parts[0].strip()
                    rest = parts[1].strip().split()
                    target = rest[0]
                    strength = float(rest[1]) if len(rest) > 1 else 0.0
                    layer_str = rest[2] if len(rest) > 2 else "L0"
                    layer = int(layer_str.replace("L", "").replace("(probe)", "").strip())
                    source = "probe" if "probe" in line else "walk"
                    edges.append(KnowledgeEdge(
                        entity=concept, relation=relation, target=target,
                        strength=strength, layer=layer, source=source))
                except (ValueError, IndexError):
                    continue
        return edges

    # ==================================================================
    # VERIFY — Did distillation actually work at the weight level?
    # ==================================================================
    def verify_knowledge(self, vindex_path: str,
                         concepts: List[str]) -> List[KnowledgeVerification]:
        """Verify whether target concepts exist in model weights.

        This is the key integration with bootcamp:
          After distillation → verify knowledge landed in weights
          If missing → retry with different training data

        Returns list of verification results.
        """
        if not self.available:
            return []

        results = []
        for concept in concepts:
            edges = self.describe(vindex_path, concept)
            if edges is None:
                results.append(KnowledgeVerification(
                    concept=concept, found=False, edges=[], confidence=0.0))
            else:
                max_strength = max((e.strength for e in edges), default=0.0)
                confidence = min(max_strength / 100.0, 1.0)  # Normalize
                results.append(KnowledgeVerification(
                    concept=concept, found=len(edges) > 0,
                    edges=edges, confidence=confidence))

        found = sum(1 for r in results if r.found)
        print(f"  🔬 Knowledge verification: {found}/{len(concepts)} concepts found")
        return results

    # ==================================================================
    # INSERT — Patch knowledge directly into weights (no retraining)
    # ==================================================================
    def insert_knowledge(self, vindex_path: str, entity: str,
                         relation: str, target: str) -> bool:
        """Insert a fact directly into the model's weight-level knowledge graph.

        This is LARQL's killer feature: edit model knowledge without retraining.
        The model will then generate text consistent with the new fact.

        Example: insert_knowledge("model.vindex", "Acme Corp", "headquarters", "London")
        """
        if not self.available:
            return False

        lql = (f'USE "{vindex_path}"; '
               f'INSERT INTO EDGES (entity, relation, target) '
               f'VALUES ("{entity}", "{relation}", "{target}");')
        result = self._run(["lql", lql])
        if result and "Inserted" in result:
            print(f"     ✅ Inserted: {entity} → {relation} → {target}")
            return True
        return False

    # ==================================================================
    # INFER — Run inference through the vindex
    # ==================================================================
    def infer(self, vindex_path: str, prompt: str,
              top_k: int = 5) -> Optional[List[Dict]]:
        """Run inference through the model's knowledge graph.

        Returns top-k predictions with probabilities.
        """
        if not self.available:
            return None

        lql = f'USE "{vindex_path}"; INFER "{prompt}" TOP {top_k};'
        result = self._run(["lql", lql])
        if result is None:
            return None

        predictions = []
        for line in result.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                try:
                    parts = line.split(".", 1)[1].strip()
                    token = parts.split("(")[0].strip()
                    prob_str = parts.split("(")[1].replace(")", "").replace("%", "").strip()
                    prob = float(prob_str) / 100.0
                    predictions.append({"token": token, "probability": prob})
                except (ValueError, IndexError):
                    continue
        return predictions

    # ==================================================================
    # DIFF — Compare two vindexes to measure knowledge gain
    # ==================================================================
    def diff_knowledge(self, vindex_before: str, vindex_after: str,
                       concepts: List[str]) -> Dict:
        """Compare knowledge before and after distillation.

        Returns a dict with gained, lost, and unchanged edges.
        """
        if not self.available:
            return {"gained": [], "lost": [], "unchanged": [], "available": False}

        before_edges = {}
        after_edges = {}

        for concept in concepts:
            b = self.describe(vindex_before, concept)
            a = self.describe(vindex_after, concept)
            if b:
                before_edges[concept] = {(e.relation, e.target) for e in b}
            if a:
                after_edges[concept] = {(e.relation, e.target) for e in a}

        gained = []
        lost = []
        unchanged = []

        all_concepts = set(list(before_edges.keys()) + list(after_edges.keys()))
        for concept in all_concepts:
            b = before_edges.get(concept, set())
            a = after_edges.get(concept, set())
            for edge in a - b:
                gained.append({"concept": concept, "relation": edge[0], "target": edge[1]})
            for edge in b - a:
                lost.append({"concept": concept, "relation": edge[0], "target": edge[1]})
            for edge in a & b:
                unchanged.append({"concept": concept, "relation": edge[0], "target": edge[1]})

        print(f"  🔬 Knowledge diff: +{len(gained)} gained, -{len(lost)} lost, "
              f"={len(unchanged)} unchanged")
        return {"gained": gained, "lost": lost, "unchanged": unchanged, "available": True}

    # ==================================================================
    # COMPILE — Repackage vindex back to safetensors/GGUF
    # ==================================================================
    def compile_to_model(self, vindex_path: str, output_path: str,
                         fmt: str = "safetensors") -> Optional[str]:
        """Compile a (possibly patched) vindex back to a deployable model.

        This closes the loop: extract → query → patch → compile → deploy.
        """
        if not self.available:
            return None

        args = ["build", vindex_path, "--compile", fmt, "--output", output_path]
        result = self._run(args, timeout=600)
        if result is not None:
            print(f"     ✅ Compiled: {vindex_path} → {output_path} ({fmt})")
            return output_path
        return None


# Singleton
_client = None

def get_larql() -> LARQLClient:
    """Get the LARQL client singleton."""
    global _client
    if _client is None:
        _client = LARQLClient()
    return _client
