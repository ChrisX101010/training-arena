"""Hardware Detector - Auto-detect GPU and apply optimal settings."""

import subprocess
import os
from typing import Dict, Optional
from pathlib import Path
import yaml


class HardwareDetector:
    def __init__(self, config_path: str = "config/hardware_profiles.yaml"):
        fp = Path(config_path)
        if fp.exists():
            with open(fp) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {"profiles": {"cpu": {"settings": {"device": "cpu", "batch_size": 1}}}}
        self.gpu_info = self._detect()
        self.profile = self._pick_profile()

    def _detect(self) -> Optional[Dict]:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split("\n")[0].split(", ")
                return {"name": parts[0].strip(), "memory_mb": int(parts[1].strip())}
        except Exception:
            pass
        return None

    def _pick_profile(self) -> Dict:
        if not self.gpu_info:
            return self.config["profiles"]["cpu"]
        vram = self.gpu_info["memory_mb"] / 1024
        p = self.config["profiles"]
        if vram >= 40:
            return p.get("high_end", p["cpu"])
        if vram >= 16:
            return p.get("mid_range", p["cpu"])
        if vram >= 8:
            return p.get("entry_level", p["cpu"])
        if vram >= 4:
            return p.get("low_end", p["cpu"])
        return p["cpu"]

    def get_settings(self, model_name: str = None) -> Dict:
        s = self.profile["settings"].copy()
        if model_name and "model_overrides" in self.config:
            s.update(self.config["model_overrides"].get(model_name, {}))
        return s

    def print_summary(self):
        print("=" * 50)
        if self.gpu_info:
            print(f"🎮 GPU: {self.gpu_info['name']} ({self.gpu_info['memory_mb']/1024:.1f} GB)")
        else:
            print("💻 CPU mode (no GPU detected)")
        print("=" * 50)
