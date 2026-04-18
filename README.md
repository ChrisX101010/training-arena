# 🏟️ Training Arena v3.0

**Self-improving LLM platform with real evaluation, Axiom Wiki integration, and TREX-inspired tree-based evolution.**
> 🔗 **[Project Page](https://ChrisX101010.github.io/training-arena/)** · [Axiom Wiki](https://github.com/abubakarsiddik31/axiom-wiki) · [Polyrating](https://github.com/eth-sri/polyrating)

Beat MiniMax M2.7 at its own game — open-source, on consumer hardware.

## What's new in v3

- **Axiom Wiki integration** — Karpathy's compounding knowledge pattern. Pre-compiled, cross-referenced wiki pages become high-quality training data. Install [axiom-wiki](https://github.com/abubakarsiddik31/axiom-wiki) as a knowledge source alongside Wikipedia and human corrections.
- **TREX-inspired tree search** — each bootcamp round is a node. If a round causes regression, the system backtracks and tries a different strategy (inspired by [TREX: Automating LLM Fine-tuning](https://arxiv.org/abs/2604.14116)).
- **Compounding memory** — insights from previous runs persist in the DB and inform future strategy.
- **2026 baselines** — Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro, MiniMax M2.7, Qwen 3.5, Llama 4, Gemma 4, DeepSeek R1.
- **Multi-source training data** — Axiom pages + human corrections + Wikipedia + teacher synthesis, all feeding into each evolution round.

## Architecture

```
Knowledge Sources                    Training Pipeline
┌──────────────┐                    ┌─────────────────────────┐
│ Axiom Wiki   │ (compiled pages)   │  BOOTCAMP               │
│ (Karpathy    │──────┐             │  TREX tree search:      │
│  pattern)    │      │             │  evaluate → diagnose →  │
├──────────────┤      │             │  synthesize → distill → │
│ Wikipedia    │──────┤──────────── │  verify → backtrack?    │
│ (real-time)  │      │             └──────────┬──────────────┘
├──────────────┤      │                        │
│ Human        │──────┘             ┌──────────┴──────────────┐
│ Corrections  │                    │  REHEARSAL               │
│ (gold std)   │                    │  Student vs Teacher      │
└──────────────┘                    │  6-dim rubric scoring    │
                                    └──────────┬──────────────┘
                                               │
                          ┌────────────────────┤
                          ▼                    ▼
                    ┌──────────┐        ┌──────────────┐
                    │ ACADEMY  │        │    ARENA     │
                    │ Domain   │        │ All models = │
                    │ speciali │        │ Polyrating   │
                    │ sts      │        │ + commercial │
                    └──────────┘        │ baselines    │
                                        └──────────────┘
```

## Quick start

```bash
pip install -r requirements.txt
pip install -e ../polyrating    # if you have the local clone

# Optional: install Axiom Wiki for compounding knowledge
npm install -g axiom-wiki
axiom-wiki init

# Bootstrap wiki
python arena.py --wiki-fetch gravity photosynthesis "Pythagorean theorem" --wiki-fetch-domain science

# Self-evolve
python arena.py --bootcamp --rounds 3

# Global leaderboard
python arena.py --arena

# Dashboard with Rock 'Em Sock 'Em
python arena.py --dashboard
```

## Axiom Wiki integration

[Axiom Wiki](https://github.com/abubakarsiddik31/axiom-wiki) implements Karpathy's compounding knowledge pattern: knowledge gets compiled once and kept current, not re-derived on every query.

Set up:
```yaml
# config/models.yaml
wiki:
  axiom_wiki_path: /path/to/your/axiom-wiki
```

Or install globally and the system detects it automatically:
```bash
npm install -g axiom-wiki
axiom-wiki init
axiom-wiki ingest your-documents/
```

Training Arena consumes Axiom's compiled pages as high-quality training data during bootcamp.

## How self-evolution works (TREX + MiniMax hybrid)

1. **Pull** all knowledge sources (Axiom + Wikipedia + corrections)
2. **Evaluate** student across domains with 6-dimension rubric
3. **Diagnose** weakest domain
4. **Synthesize** targeted training data from multiple sources
5. **Distill** student on that data
6. **Verify** — did it improve? If regression → **backtrack** and try different strategy
7. **Rehearsal** — student vs teacher head-to-head with real rubric scoring
8. **Arena** — compete against the field

## Commercial baselines (2026)

The arena includes reference ELO scores from published benchmarks:

| Model | ELO | Type |
|-------|-----|------|
| Claude Opus 4.6 | 2088 | Frontier |
| GPT-5.4 | 2074 | Frontier |
| Gemini 3.1 Pro | 2046 | Frontier |
| MiniMax M2.7 | 2032 | Self-evolving |
| Qwen 3.5 9B | 1948 | Open-source |
| Llama 4 Scout | 1920 | Open-source |
| Gemma 4 9B | 1906 | Open-source |

Your self-evolved models compete against these baselines on the leaderboard.

## CLI

```
python arena.py --bootcamp     # Self-evolve (TREX tree search)
python arena.py --academy      # Train domain specialists
python arena.py --arena        # Global leaderboard
python arena.py --rehearsal    # Teacher vs student benchmark
python arena.py --dashboard    # Rock 'Em Sock 'Em UI
python arena.py --wiki-fetch   # Pull Wikipedia data
python arena.py --list-models  # Show configured models
```

## Hardware

Tested on RTX 3050 4GB. Auto-detects GPU and applies optimal profile.

## Credits

- [Polyrating](https://github.com/eth-sri/polyrating) — ETH-SRI's bias-aware rating system
- [Axiom Wiki](https://github.com/abubakarsiddik31/axiom-wiki) — Karpathy's compounding knowledge pattern
- [TREX](https://arxiv.org/abs/2604.14116) — tree-based exploration for LLM training
- [MiniMax M2.7](https://www.minimax.io/models/text/m27) — recursive self-optimization inspiration
- [NVIDIA QCalEval](https://github.com/nvidia/QCalEval) — 6-dimension rubric methodology

## License

MIT
