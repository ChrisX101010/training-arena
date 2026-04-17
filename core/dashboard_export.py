"""Simple data export utility - dumps DB + wiki state to JSON for the React dashboard."""

import json
from pathlib import Path
from core.metrics_db import MetricsDatabase
from core.llm_wiki import LLMWiki


def export_dashboard_data(output_path: str = "./results/dashboard_data.json"):
    """Export everything the React dashboard needs in one JSON file."""
    db = MetricsDatabase()
    wiki = LLMWiki()

    data = {
        "leaderboard": db.get_global_rankings(),
        "recent_matches": db.get_recent_matches(20),
        "evolution_log": db.get_evolution_log(limit=20),
        "training_runs": db.get_training_history(20),
        "wiki_stats": wiki.get_wiki_stats(),
        "wiki_commits": wiki.get_commit_log(10),
        "corrections": wiki.get_corrections()[-10:],
    }

    # Add commercial baselines
    try:
        from core.arena import COMMERCIAL_BASELINES
        if data["leaderboard"]:
            max_r = max(r["elo_rating"] for r in data["leaderboard"])
            for name, scores in COMMERCIAL_BASELINES.items():
                avg = sum(scores.values()) / len(scores)
                data["leaderboard"].append({
                    "model_name": name,
                    "elo_rating": round(1500 + (avg - 0.5) * 1400, 1),
                    "wins": 0, "losses": 0, "draws": 0, "type": "commercial",
                })
            data["leaderboard"].sort(key=lambda x: x["elo_rating"], reverse=True)
    except Exception:
        pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✅ Exported dashboard data → {output_path}")
    return data


if __name__ == "__main__":
    export_dashboard_data()
