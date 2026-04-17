"""Model Hub - Centralized model loading and inference."""

import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, List


class ModelHub:
    """Single entry point for all model operations (Ollama + HuggingFace)."""

    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._cache: Dict[str, dict] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ollama_ok = self._check_ollama()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def get_available_models(self) -> List[str]:
        return [m["name"] for m in self.config["models"]]

    def get_model_info(self, name: str) -> dict:
        for m in self.config["models"]:
            if m["name"] == name:
                return m
        raise ValueError(f"Model '{name}' not in config")

    def get_models_by_role(self, role: str) -> List[str]:
        return [m["name"] for m in self.config["models"] if m.get("role") == role]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_model(self, name: str) -> dict:
        if name in self._cache:
            return self._cache[name]

        info = self.get_model_info(name)

        if info["provider"] == "ollama":
            if not self.ollama_ok:
                raise RuntimeError("Ollama not running – start with `ollama serve`")
            self._cache[name] = {"type": "ollama", "name": name}
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(name)
            tokenizer.pad_token = tokenizer.eos_token
            self._cache[name] = {
                "type": "huggingface",
                "model": model,
                "tokenizer": tokenizer,
            }

        return self._cache[name]

    def unload(self, name: str):
        if name in self._cache:
            del self._cache[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate_response(self, name: str, prompt: str, max_tokens: int = 100) -> str:
        entry = self.load_model(name)

        if entry["type"] == "ollama":
            import ollama
            resp = ollama.generate(
                model=name, prompt=prompt,
                options={"num_predict": max_tokens},
            )
            return resp["response"]

        model = entry["model"]
        tokenizer = entry["tokenizer"]
        inputs = tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_ollama() -> bool:
        try:
            import ollama
            ollama.list()
            return True
        except Exception:
            return False
