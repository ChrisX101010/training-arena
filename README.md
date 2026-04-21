# 🏟️ Training Arena v3.0

**Self-improving LLM platform with real evaluation, PEFT/LoRA training, LARQL weight verification, and TREX-inspired tree-based evolution.**

> 🔗 **[Project Page](https://chrisx101010.github.io/training-arena/)** · [Axiom Wiki](https://github.com/abubakarsiddik31/axiom-wiki) · [Polyrating](https://github.com/eth-sri/polyrating) · [LARQL](https://github.com/chrishayuk/larql)

## Architecture

```
Knowledge Sources                    Training Pipeline
┌──────────────┐                    ┌──────────────────────────────┐
│ Axiom Wiki   │ (compiled pages)   │  BOOTCAMP (TREX tree search) │
│ (Karpathy    │──────┐             │  evaluate → diagnose →       │
│  pattern)    │      │             │  synthesize → distill(LoRA) →│
├──────────────┤      │             │  LARQL verify → backtrack?   │
│ Wikipedia    │──────┤──────────── └──────────────┬───────────────┘
│ (real-time)  │      │                            │
├──────────────┤      │             ┌──────────────┴───────────────┐
│ Human        │──────┘             │  REHEARSAL (student vs teacher│
│ Corrections  │                    │  6-dimension rubric scoring)  │
└──────────────┘                    └──────────────┬───────────────┘
                                                   │
                          ┌────────────────────────┤
                          ▼                        ▼
                    ┌──────────┐        ┌───────────────────┐
                    │ ACADEMY  │        │      ARENA        │
                    │ Domain   │        │ ELO + Polyrating  │
                    │ speciali │        │ + 2026 commercial │
                    │ sts      │        │ baselines         │
                    └──────────┘        └───────────────────┘
```

## What's in v3

- **PEFT/LoRA training** — trains ~1% of parameters, fits on 4GB VRAM. Auto-detects target modules per architecture (GPT-2: c_attn/c_proj, Qwen/Llama: q_proj/k_proj/v_proj)
- **LARQL integration** — weight-level knowledge verification after distillation. Instead of scoring text output, verifies knowledge exists in the model's weights. Like scanning a brain vs giving a quiz
- **Axiom Wiki** — Karpathy's compounding knowledge pattern as a training data source
- **TREX tree search** — backtracks when evolution rounds cause regression
- **7-signal heuristic scorer** — differentiates model outputs when no LLM judge available
- **2026 baselines** — Claude 4.6, GPT-5.4, MiniMax M2.7, Qwen 3.5, Gemma 4, DeepSeek R1
- **Rock 'Em Sock 'Em dashboard** — RED ROCKER vs BLUE BOMBER with authentic animations

## Quick start

```bash
git clone https://github.com/ChrisX101010/training-arena.git
cd training-arena
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run arena (real scores, not 0.500 draws)
python arena.py --arena

# Self-evolve with LoRA
python arena.py --bootcamp --rounds 3

# Dashboard
python arena.py --dashboard
```

## LARQL — Weight-Level Knowledge Verification

[LARQL](https://github.com/chrishayuk/larql) decompiles transformer weights into a queryable knowledge graph. After distillation, Training Arena can verify whether knowledge actually landed in the model's weights — not just whether the text output looks right.

```
Bootcamp distill (LoRA) → larql extract-index → DESCRIBE target concepts
  → knowledge edge found in weights? → training verified ✅
  → knowledge missing from weights? → backtrack, retry 🔄
```

Install (requires Rust): `cargo install larql`

When LARQL is not installed, the system gracefully falls back to text-based heuristic scoring. All LARQL methods return None when unavailable — zero crashes, zero extra dependencies.

### How it works

```python
from core.larql_integration import get_larql

larql = get_larql()

# Extract model into queryable vindex
larql.extract_index("distilgpt2", "student.vindex")

# Verify knowledge at weight level
results = larql.verify_knowledge("student.vindex", ["France", "gravity", "Python"])

# Patch knowledge directly (no retraining!)
larql.insert_knowledge("student.vindex", "Acme Corp", "headquarters", "London")

# Compile back to deployable model
larql.compile_to_model("student.vindex", "patched-model/", fmt="safetensors")
```

## PEFT/LoRA Training

Full distillation updates all model parameters and needs 8GB+ VRAM. LoRA trains adapters on ~1% of parameters, fitting comfortably on 4GB.

The trainer auto-detects the correct target modules per architecture:

| Architecture | LoRA Targets |
|---|---|
| GPT-2, DistilGPT-2 | c_attn, c_proj |
| Qwen 2.5/3.5, Llama 3/4 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Phi-3 | q_proj, k_proj, v_proj, dense, fc1, fc2 |
| Gemma 4 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

QLoRA goes further — quantizes the base model to 4-bit, then trains LoRA adapters on top. Uses ~2GB VRAM for a 0.5B model.

## Commercial baselines (2026)

| Model | ELO | Type |
|-------|-----|------|
| Claude Opus 4.6 | 2088 | Frontier |
| GPT-5.4 | 2074 | Frontier |
| Gemini 3.1 Pro | 2046 | Frontier |
| MiniMax M2.7 | 2032 | Self-evolving |
| Qwen 3.5 9B | 1948 | Open-source |
| Llama 4 Scout | 1920 | Open-source |
| Gemma 4 9B | 1906 | Open-source |
| DeepSeek R1 | 1886 | Open-source |

## Planned integrations

- **[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** — Google's KV cache compression (6x, ICLR 2026). When pip-installable, enables Qwen3.5-4B on 4GB VRAM
- **[Agentic Stack](https://github.com/codejunkie99/agentic-stack)** — portable 4-layer memory pattern with dream cycle. Core concept implemented as compounding_memory in DB

## Hardware

Tested on NVIDIA RTX 3050 Laptop GPU (4GB VRAM). Auto-detects GPU and applies optimal profile.

## CLI

```
python arena.py --bootcamp     # Self-evolve (TREX + LoRA + LARQL verify)
python arena.py --academy      # Train domain specialists
python arena.py --arena        # Global leaderboard
python arena.py --rehearsal    # Teacher vs student benchmark
python arena.py --dashboard    # Rock 'Em Sock 'Em UI
python arena.py --wiki-fetch   # Pull Wikipedia data
python arena.py --list-models  # Show configured models
```

## Credits

- [LARQL](https://github.com/chrishayuk/larql) — queryable transformer knowledge graphs (Lazarus)
- [Polyrating](https://github.com/eth-sri/polyrating) — ETH-SRI bias-aware rating
- [Axiom Wiki](https://github.com/abubakarsiddik31/axiom-wiki) — Karpathy compounding knowledge
- [TREX](https://arxiv.org/abs/2604.14116) — tree-based exploration for LLM training
- [Agentic Stack](https://github.com/codejunkie99/agentic-stack) — portable agent memory patterns
- [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — KV cache compression
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — self-improving skill patterns
- [MiniMax M2.7](https://www.minimax.io) — recursive self-optimization inspiration

## License

MIT
