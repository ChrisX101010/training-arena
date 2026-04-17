"""LLM Wiki v3 - Multi-source compounding knowledge engine.

Knowledge sources (all feed into training):
  1. AXIOM WIKI   — Karpathy-pattern persistent wiki via axiom-wiki CLI
  2. WIKIPEDIA    — validated real-time web data
  3. HUMAN CURATION — expert corrections to bad model outputs
  4. LOCAL ARTICLES — manually created markdown articles

Inspired by Karpathy's LLM Wiki pattern: knowledge is compiled once
and kept current, not re-derived on every query. Axiom Wiki handles
the compounding (ingest → cross-reference → lint), we consume its
compiled pages as high-quality training data.

The wiki is the ground truth that makes distillation actually work.
"""

import os
import json
import hashlib
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import yaml

try:
    import urllib.request
    WEB_OK = True
except ImportError:
    WEB_OK = False

try:
    import git
    GIT_OK = True
except ImportError:
    GIT_OK = False

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    VECTOR_OK = True
except ImportError:
    VECTOR_OK = False


class LLMWiki:
    """Multi-source compounding knowledge engine."""

    DOMAINS = ["general", "math", "science", "coding", "creative", "history", "legal", "medical"]

    TRUSTED_SOURCES = {
        "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/",
    }

    def __init__(self, wiki_path: str = "./llm_wiki", remote_url: str = "",
                 axiom_wiki_path: str = None):
        self.wiki_path = Path(wiki_path)
        self.wiki_path.mkdir(parents=True, exist_ok=True)
        self.remote_url = remote_url
        self.axiom_path = axiom_wiki_path

        # Detect axiom-wiki
        self.axiom_available = self._check_axiom()

        # Git
        self.repo = None
        if GIT_OK:
            if not (self.wiki_path / ".git").exists():
                self.repo = git.Repo.init(self.wiki_path)
                readme = self.wiki_path / "README.md"
                readme.write_text("# LLM Wiki\n\nMulti-source compounding knowledge base.\n")
                self.repo.index.add(["README.md"])
                self.repo.index.commit("init: LLM Wiki v3")
            else:
                self.repo = git.Repo(self.wiki_path)
            if remote_url:
                try:
                    self.repo.create_remote("origin", remote_url)
                except Exception:
                    pass

        for d in self.DOMAINS:
            (self.wiki_path / d).mkdir(exist_ok=True)
        (self.wiki_path / "_corrections").mkdir(exist_ok=True)
        (self.wiki_path / "_axiom").mkdir(exist_ok=True)

        # Vector search
        self.collections: Dict = {}
        if VECTOR_OK:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.chroma = chromadb.PersistentClient(path=str(self.wiki_path / ".chroma"))
            for d in self.DOMAINS:
                try:
                    self.collections[d] = self.chroma.get_collection(f"wiki_{d}")
                except Exception:
                    self.collections[d] = self.chroma.create_collection(f"wiki_{d}")

    # ==================================================================
    # AXIOM WIKI INTEGRATION (Karpathy's compounding pattern)
    # ==================================================================
    def _check_axiom(self) -> bool:
        """Check if axiom-wiki CLI is available."""
        try:
            r = subprocess.run(["axiom-wiki", "status"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                print("📚 Axiom Wiki connected (Karpathy pattern)")
                return True
        except Exception:
            pass

        # Also check if axiom wiki directory exists with pages
        if self.axiom_path and Path(self.axiom_path).exists():
            wiki_dir = Path(self.axiom_path) / "wiki"
            if wiki_dir.exists() and list(wiki_dir.glob("*.md")):
                print(f"📚 Axiom Wiki found at {self.axiom_path}")
                return True
        return False

    def ingest_from_axiom(self, query: str = None, domain: str = "general",
                          max_pages: int = 20) -> int:
        """
        Pull compiled knowledge from Axiom Wiki into our training pipeline.

        Axiom Wiki maintains pre-synthesized, cross-referenced pages —
        much higher quality than raw document chunks. This is Karpathy's
        key insight: compiled knowledge > retrieved fragments.
        """
        ingested = 0

        # Method 1: Use axiom-wiki CLI query
        if self.axiom_available:
            try:
                cmd = ["axiom-wiki", "query", query or f"key concepts in {domain}",
                       "--format", "json"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if r.returncode == 0 and r.stdout.strip():
                    try:
                        data = json.loads(r.stdout)
                        result_text = data.get("result", r.stdout)
                    except json.JSONDecodeError:
                        result_text = r.stdout.strip()

                    if len(result_text) > 50:
                        slug = re.sub(r'[^a-z0-9_]', '_', (query or domain).lower().strip())[:60]
                        filepath = self.wiki_path / "_axiom" / f"{slug}.md"
                        filepath.write_text(
                            f"---\nsource: axiom-wiki\nquery: {query}\n"
                            f"domain: {domain}\nfetched: {datetime.now().isoformat()}\n---\n"
                            f"{result_text}\n"
                        )
                        ingested += 1
            except Exception as e:
                print(f"    ⚠️ Axiom CLI query failed: {e}")

        # Method 2: Read directly from axiom wiki directory
        if self.axiom_path:
            wiki_dir = Path(self.axiom_path) / "wiki"
            if wiki_dir.exists():
                for md_file in sorted(wiki_dir.glob("*.md"))[:max_pages]:
                    try:
                        content = md_file.read_text()
                        if len(content) > 100:
                            dest = self.wiki_path / "_axiom" / md_file.name
                            if not dest.exists() or dest.read_text() != content:
                                dest.write_text(content)
                                ingested += 1
                    except Exception:
                        pass

        if ingested > 0:
            print(f"    📚 Ingested {ingested} pages from Axiom Wiki")
            if GIT_OK and self.repo:
                self.repo.git.add(A=True)
                self.repo.index.commit(f"axiom-ingest: {ingested} pages for {domain}")

        return ingested

    def sync_axiom(self) -> str:
        """Sync axiom wiki (run its ingest + lint cycle)."""
        if not self.axiom_available:
            return "Axiom Wiki not available. Install: npm install -g axiom-wiki"
        try:
            r = subprocess.run(["axiom-wiki", "sync"], capture_output=True, text=True, timeout=60)
            return r.stdout.strip() or "Synced"
        except Exception as e:
            return f"Sync failed: {e}"

    # ==================================================================
    # WIKIPEDIA FETCH
    # ==================================================================
    def fetch_from_wikipedia(self, topic: str, domain: str = "general") -> Optional[str]:
        if not WEB_OK:
            return None
        slug = topic.replace(" ", "_")
        url = self.TRUSTED_SOURCES["wikipedia"] + slug
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TrainingArena/3.0 (https://github.com/ChrisX101010/training-arena)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            if "extract" not in data or len(data["extract"]) < 20:
                return None
            content = data["extract"]
            title = data.get("title", topic)
            source_url = data.get("content_urls", {}).get("desktop", {}).get("page", url)
            path = self.create_article(title, domain, content, "wikipedia-fetch", [source_url])
            print(f"    📰 Fetched: {title} ({len(content)} chars)")
            return path
        except Exception as e:
            print(f"    ⚠️ Wikipedia fetch failed: {e}")
            return None

    def fetch_batch(self, topics: List[str], domain: str = "general") -> int:
        fetched = sum(1 for t in topics if self.fetch_from_wikipedia(t, domain))
        if fetched and GIT_OK and self.repo:
            self.repo.git.add(A=True)
            self.repo.index.commit(f"web-fetch: {fetched} articles for {domain}")
        return fetched

    # ==================================================================
    # HUMAN CURATION
    # ==================================================================
    def record_correction(self, prompt: str, bad_output: str,
                          corrected_output: str, domain: str = "general",
                          author: str = "human-expert") -> str:
        correction = {
            "prompt": prompt, "bad_output": bad_output,
            "corrected_output": corrected_output, "domain": domain,
            "author": author, "timestamp": datetime.now().isoformat(),
        }
        cid = hashlib.md5(f"{prompt}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        filepath = self.wiki_path / "_corrections" / f"{cid}.json"
        filepath.write_text(json.dumps(correction, indent=2))
        if GIT_OK and self.repo:
            self.repo.index.add([str(filepath.relative_to(self.wiki_path))])
            self.repo.index.commit(f"correction: {domain} by {author}")
        if VECTOR_OK and domain in self.collections:
            emb = self.encoder.encode(corrected_output).tolist()
            self.collections[domain].upsert(
                documents=[f"Q: {prompt}\nA: {corrected_output}"],
                embeddings=[emb],
                metadatas=[{"title": f"correction-{cid}", "author": author, "path": str(filepath)}],
                ids=[cid])
        return str(filepath)

    def get_corrections(self, domain: str = None) -> List[Dict]:
        corrections = []
        for fp in (self.wiki_path / "_corrections").glob("*.json"):
            try:
                c = json.loads(fp.read_text())
                if domain is None or c.get("domain") == domain:
                    corrections.append(c)
            except Exception:
                pass
        return corrections

    def get_training_data_from_corrections(self, domain: str = None) -> List[str]:
        return [f"Question: {c['prompt']}\nAnswer: {c['corrected_output']}"
                for c in self.get_corrections(domain)]

    # ==================================================================
    # ARTICLES CRUD
    # ==================================================================
    def create_article(self, title: str, domain: str, content: str,
                       author: str = "system", sources: List[str] = None) -> str:
        if domain not in self.DOMAINS:
            raise ValueError(f"Domain must be one of {self.DOMAINS}")
        slug = re.sub(r'[^a-z0-9_]', '_', title.lower().strip())
        filepath = self.wiki_path / domain / f"{slug}.md"
        meta = {"title": title, "domain": domain, "author": author,
                "created": datetime.now().isoformat(), "version": 1,
                "sources": sources or []}
        filepath.write_text(f"---\n{yaml.dump(meta)}---\n{content}\n")
        if GIT_OK and self.repo:
            self.repo.index.add([str(filepath.relative_to(self.wiki_path))])
            self.repo.index.commit(f"add: {title} [{domain}] by {author}")
        if VECTOR_OK and domain in self.collections:
            aid = hashlib.md5(f"{domain}/{title}".encode()).hexdigest()
            emb = self.encoder.encode(content).tolist()
            self.collections[domain].upsert(
                documents=[content], embeddings=[emb],
                metadatas=[{"title": title, "author": author, "path": str(filepath)}],
                ids=[aid])
        return str(filepath)

    def get_article(self, title: str, domain: str) -> Optional[Dict]:
        slug = re.sub(r'[^a-z0-9_]', '_', title.lower().strip())
        fp = self.wiki_path / domain / f"{slug}.md"
        if not fp.exists():
            return None
        raw = fp.read_text()
        meta, body = {}, raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    pass
                body = parts[2].strip()
        return {"title": title, "domain": domain, "frontmatter": meta,
                "content": body, "path": str(fp)}

    def search(self, query: str, domain: str = "general", n: int = 5) -> List[Dict]:
        if not VECTOR_OK or domain not in self.collections:
            return []
        emb = self.encoder.encode(query).tolist()
        res = self.collections[domain].query(query_embeddings=[emb], n_results=n,
                                              include=["documents", "metadatas", "distances"])
        return [{"content": doc, "title": meta["title"], "path": meta["path"],
                 "relevance": round(1 - dist, 3)}
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    # ==================================================================
    # TRAINING DATA GENERATION — All sources combined
    # ==================================================================
    def generate_training_prompts(self, domain: str, n: int = 10) -> List[str]:
        """
        Compile training data from ALL sources:
        1. Axiom Wiki pages (highest quality — pre-compiled, cross-referenced)
        2. Human corrections (gold standard — expert-validated)
        3. Local wiki articles (manually curated)
        4. Wikipedia fetches (validated web data)

        This is the Karpathy compounding pattern in action:
        each source makes the training data richer.
        """
        prompts = []

        # 1. Axiom Wiki pages (best quality — pre-synthesized)
        axiom_dir = self.wiki_path / "_axiom"
        for md in sorted(axiom_dir.glob("*.md"))[:5]:
            try:
                content = md.read_text()
                # Strip frontmatter
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        content = parts[2].strip()
                if len(content) > 50:
                    # Use axiom content as comprehensive answers
                    title = md.stem.replace("_", " ")
                    prompts.append(f"Question: Explain {title} comprehensively.\nAnswer: {content[:500]}")
            except Exception:
                pass

        # 2. Human corrections (gold standard)
        prompts.extend(self.get_training_data_from_corrections(domain))

        # 3. Local articles
        dp = self.wiki_path / domain
        for md in list(dp.glob("*.md"))[:5]:
            a = self.get_article(md.stem.replace("_", " "), domain)
            if a and a["content"] and len(a["content"]) > 30:
                prompts.append(f"Question: Explain what {a['title']} is.\nAnswer: {a['content'][:300]}")

        return prompts[:n]

    # ==================================================================
    # GIT SYNC
    # ==================================================================
    def pull(self) -> str:
        if not (GIT_OK and self.repo and self.remote_url):
            return "No remote configured"
        try:
            self.repo.remotes.origin.pull()
            return "Pulled latest"
        except Exception as e:
            return f"Pull failed: {e}"

    def push(self, message: str = "wiki update") -> str:
        if not (GIT_OK and self.repo and self.remote_url):
            return "No remote configured"
        try:
            self.repo.git.add(A=True)
            self.repo.index.commit(message)
            self.repo.remotes.origin.push()
            return "Pushed"
        except Exception as e:
            return f"Push failed: {e}"

    def get_wiki_stats(self) -> Dict:
        stats = {"total_articles": 0, "domains": {}, "corrections": 0,
                 "axiom_pages": 0, "last_commit": None}
        for d in self.DOMAINS:
            count = len(list((self.wiki_path / d).glob("*.md")))
            stats["domains"][d] = count
            stats["total_articles"] += count
        stats["corrections"] = len(list((self.wiki_path / "_corrections").glob("*.json")))
        stats["axiom_pages"] = len(list((self.wiki_path / "_axiom").glob("*.md")))
        stats["axiom_available"] = self.axiom_available
        if GIT_OK and self.repo:
            try:
                c = self.repo.head.commit
                stats["last_commit"] = {"message": c.message.strip(),
                                        "date": datetime.fromtimestamp(c.committed_date).isoformat(),
                                        "author": str(c.author)}
            except Exception:
                pass
        return stats

    def get_commit_log(self, limit: int = 20) -> List[Dict]:
        if not (GIT_OK and self.repo):
            return []
        return [{"hash": c.hexsha[:8], "message": c.message.strip(),
                 "date": datetime.fromtimestamp(c.committed_date).isoformat(),
                 "author": str(c.author)}
                for c in self.repo.iter_commits(max_count=limit)]
