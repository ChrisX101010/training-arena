#!/usr/bin/env python3
"""Training Arena — Self-Improving LLM Platform."""

import sys, json, yaml, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.model_hub import ModelHub
from core.curriculum_engine import CurriculumEngine
from core.llm_wiki import LLMWiki

BANNER = """
╔══════════════════════════════════════════════════╗
║  🏟️  TRAINING ARENA  v2.0                       ║
║  Self-Evolve · Specialise · Compete              ║
╚══════════════════════════════════════════════════╝"""


def main():
    p = argparse.ArgumentParser(description="Training Arena")
    p.add_argument("--config", default="config/models.yaml")

    # Modes
    p.add_argument("--bootcamp", action="store_true",
                   help="Self-evolve a model (ends with rehearsal benchmark)")
    p.add_argument("--academy", action="store_true",
                   help="Train domain specialists")
    p.add_argument("--arena", action="store_true",
                   help="Global leaderboard tournament")
    p.add_argument("--rehearsal", action="store_true",
                   help="Run rehearsal benchmark only (teacher vs student)")
    p.add_argument("--full", action="store_true",
                   help="Bootcamp + Academy + Arena")
    p.add_argument("--dashboard", action="store_true",
                   help="Launch Streamlit UI")
    p.add_argument("--list-models", action="store_true")

    # Options
    p.add_argument("--teacher")
    p.add_argument("--student")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--domains", nargs="+", default=["math", "science", "coding", "creative"])
    p.add_argument("--no-rehearsal", action="store_true")
    p.add_argument("--no-baselines", action="store_true", help="Skip commercial baselines in Arena")

    # Wiki
    p.add_argument("--wiki-stats", action="store_true")
    p.add_argument("--wiki-pull", action="store_true")
    p.add_argument("--wiki-push", action="store_true")
    p.add_argument("--wiki-fetch", nargs="+", metavar="TOPIC",
                   help="Fetch Wikipedia articles into the wiki")
    p.add_argument("--wiki-fetch-domain", default="general")
    p.add_argument("--create-article", nargs=3, metavar=("TITLE", "DOMAIN", "CONTENT"))

    args = p.parse_args()
    print(BANNER)

    if args.dashboard:
        import subprocess
        subprocess.run(["streamlit", "run", str(Path(__file__).parent / "dashboard.py")])
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.list_models:
        print("\n📋 Models:")
        for m in config["models"]:
            on = "✓" if m.get("enabled", True) else "✗"
            print(f"  {on} {m['name']}  ({m['provider']})  [{m.get('role', '-')}]")
        return

    # Wiki
    wiki = LLMWiki(config.get("wiki", {}).get("path", "./llm_wiki"),
                    config.get("wiki", {}).get("remote_url", ""))
    if args.wiki_stats:
        s = wiki.get_wiki_stats()
        print(f"\n📚 Wiki: {s['total_articles']} articles, {s['corrections']} corrections")
        for d, c in s["domains"].items():
            if c: print(f"   {d}: {c}")
        return
    if args.wiki_pull: print(wiki.pull()); return
    if args.wiki_push: print(wiki.push()); return
    if args.wiki_fetch:
        count = wiki.fetch_batch(args.wiki_fetch, args.wiki_fetch_domain)
        print(f"✅ Fetched {count} articles into {args.wiki_fetch_domain}")
        return
    if args.create_article:
        t, d, c = args.create_article
        print(f"✅ {wiki.create_article(t, d, c)}"); return

    # Pipeline
    engine = CurriculumEngine(args.config)

    if args.bootcamp:
        engine.run_bootcamp(args.student, args.teacher, args.rounds,
                            run_rehearsal=not args.no_rehearsal)
        return
    if args.academy:
        engine.run_academy(args.domains, args.teacher, args.student); return
    if args.arena:
        rankings = engine.run_arena(include_baselines=not args.no_baselines)
        print("\n🏆 Global Leaderboard:")
        for i, r in enumerate(rankings, 1):
            tag = f" [{r.get('type', 'local')}]"
            print(f"  {i}. {r['model_name']}{tag}  {r['elo_rating']:.0f}  "
                  f"W{r['wins']} L{r['losses']} D{r['draws']}")
        return
    if args.rehearsal:
        from core.rehearsal_gym import RehearsalGym
        hub = ModelHub(args.config)
        gym = RehearsalGym(hub)
        student = args.student or config["training"].get("student_model", "distilgpt2")
        teacher = args.teacher or "Qwen/Qwen2.5-0.5B"
        with open(config["arena"]["prompts_file"]) as f:
            prompts = [p["text"] for p in json.load(f)["prompts"]][:10]
        gym.benchmark_and_save(student, teacher, prompts); return
    if args.full:
        engine.run_full(args.student, args.teacher, args.rounds, args.domains); return

    print("""Usage:
  python arena.py --bootcamp     Self-evolve a model (with rehearsal benchmark)
  python arena.py --academy      Train domain specialists
  python arena.py --arena        Global leaderboard tournament
  python arena.py --rehearsal    Teacher/Student benchmark only
  python arena.py --full         All three pipelines
  python arena.py --dashboard    Launch UI

  python arena.py --wiki-fetch gravity photosynthesis    Pull Wikipedia
  python arena.py --wiki-stats                           Wiki info""")


if __name__ == "__main__":
    main()
