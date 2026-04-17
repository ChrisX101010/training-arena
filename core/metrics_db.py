"""Metrics Database v3 - With tree tracking and compounding memory."""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional


class MetricsDatabase:
    def __init__(self, db_path: str = "./results/metrics.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL, domain TEXT NOT NULL,
                metric_type TEXT NOT NULL, score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS head_to_head (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_a TEXT NOT NULL, model_b TEXT NOT NULL,
                winner TEXT, prompt TEXT, domain TEXT,
                score_a REAL, score_b REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS domain_elo (
                model_name TEXT NOT NULL, domain TEXT NOT NULL,
                elo_rating REAL DEFAULT 1500,
                wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_name, domain)
            );
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_model TEXT, student_model TEXT,
                phase TEXT, round_num INTEGER,
                prompts_used INTEGER, output_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS self_evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL, iteration INTEGER,
                gap_identified TEXT, improvement_delta REAL,
                synthetic_prompts_generated INTEGER,
                strategy TEXT DEFAULT 'default',
                backtracked INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS compounding_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_scores_model ON model_scores(model_name);
            CREATE INDEX IF NOT EXISTS idx_h2h_time ON head_to_head(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory ON compounding_memory(model_name, key);
        """)
        self.conn.commit()

    # Recording
    def record_score(self, model_name, domain, metric_type, score, metadata=None):
        self.conn.execute(
            "INSERT INTO model_scores (model_name, domain, metric_type, score, metadata) VALUES (?,?,?,?,?)",
            (model_name, domain, metric_type, score, json.dumps(metadata or {})))
        self.conn.commit()

    def record_match(self, model_a, model_b, winner, prompt, domain, score_a, score_b):
        self.conn.execute(
            "INSERT INTO head_to_head (model_a, model_b, winner, prompt, domain, score_a, score_b) VALUES (?,?,?,?,?,?,?)",
            (model_a, model_b, winner, prompt, domain, score_a, score_b))
        self._upsert_elo(model_a, model_b, winner, domain)
        self.conn.commit()

    def record_training_run(self, teacher, student, phase, round_num, prompts_used, output_path):
        self.conn.execute(
            "INSERT INTO training_runs (teacher_model, student_model, phase, round_num, prompts_used, output_path) VALUES (?,?,?,?,?,?)",
            (teacher, student, phase, round_num, prompts_used, output_path))
        self.conn.commit()

    def record_evolution_step(self, model_name, iteration, gap, delta, synth_count,
                              strategy="default", backtracked=False):
        self.conn.execute(
            "INSERT INTO self_evolution_log (model_name, iteration, gap_identified, improvement_delta, "
            "synthetic_prompts_generated, strategy, backtracked) VALUES (?,?,?,?,?,?,?)",
            (model_name, iteration, gap, delta, synth_count, strategy, int(backtracked)))
        self.conn.commit()

    # Compounding memory (persists insights across runs)
    def remember(self, model_name: str, key: str, value: str, session_id: str = ""):
        self.conn.execute(
            "INSERT INTO compounding_memory (model_name, key, value, session_id) VALUES (?,?,?,?)",
            (model_name, key, value, session_id))
        self.conn.commit()

    def recall(self, model_name: str, key: str = None) -> List[Dict]:
        if key:
            rows = self.conn.execute(
                "SELECT * FROM compounding_memory WHERE model_name=? AND key=? ORDER BY timestamp DESC",
                (model_name, key)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM compounding_memory WHERE model_name=? ORDER BY timestamp DESC LIMIT 50",
                (model_name,)).fetchall()
        return [dict(r) for r in rows]

    # ELO
    def _upsert_elo(self, a, b, winner, domain, K=32):
        def _get(model):
            r = self.conn.execute(
                "SELECT elo_rating, wins, losses, draws FROM domain_elo WHERE model_name=? AND domain=?",
                (model, domain)).fetchone()
            return dict(r) if r else {"elo_rating": 1500, "wins": 0, "losses": 0, "draws": 0}
        da, db = _get(a), _get(b)
        ea = 1 / (1 + 10 ** ((db["elo_rating"] - da["elo_rating"]) / 400))
        if winner == a:
            sa, sb = 1, 0; da["wins"] += 1; db["losses"] += 1
        elif winner == b:
            sa, sb = 0, 1; da["losses"] += 1; db["wins"] += 1
        else:
            sa, sb = 0.5, 0.5; da["draws"] += 1; db["draws"] += 1
        da["elo_rating"] += K * (sa - ea)
        db["elo_rating"] += K * ((1 - sa) - (1 - ea))
        for model, d in [(a, da), (b, db)]:
            self.conn.execute(
                "INSERT OR REPLACE INTO domain_elo (model_name, domain, elo_rating, wins, losses, draws, last_updated) "
                "VALUES (?,?,?,?,?,?, CURRENT_TIMESTAMP)", (model, domain, d["elo_rating"], d["wins"], d["losses"], d["draws"]))

    # Queries
    def get_domain_rankings(self, domain):
        return [dict(r) for r in self.conn.execute(
            "SELECT model_name, elo_rating, wins, losses, draws FROM domain_elo WHERE domain=? ORDER BY elo_rating DESC",
            (domain,)).fetchall()]

    def get_global_rankings(self):
        return [dict(r) for r in self.conn.execute(
            "SELECT model_name, AVG(elo_rating) AS elo_rating, SUM(wins) AS wins, SUM(losses) AS losses, SUM(draws) AS draws "
            "FROM domain_elo GROUP BY model_name ORDER BY elo_rating DESC").fetchall()]

    def get_recent_matches(self, limit=50):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM head_to_head ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()]

    def get_training_history(self, limit=20):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM training_runs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()]

    def get_evolution_log(self, model_name=None, limit=20):
        if model_name:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM self_evolution_log WHERE model_name=? ORDER BY timestamp DESC LIMIT ?",
                (model_name, limit)).fetchall()]
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM self_evolution_log ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()]

    def recommend_model(self, domain, criteria="elo"):
        row = self.conn.execute(
            "SELECT model_name FROM domain_elo WHERE domain=? ORDER BY elo_rating DESC LIMIT 1",
            (domain,)).fetchone()
        return row["model_name"] if row else None

    def get_category_breakdown(self, model_name):
        return [dict(r) for r in self.conn.execute(
            "SELECT domain AS category, COUNT(CASE WHEN winner=? THEN 1 END) AS wins, COUNT(*) AS total "
            "FROM head_to_head WHERE model_a=? OR model_b=? GROUP BY domain",
            (model_name, model_name, model_name)).fetchall()]
