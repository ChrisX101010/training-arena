"""Trainer - Knowledge distillation from teacher to student model."""

import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from core.model_hub import ModelHub


class DistillationTrainer:
    def __init__(self, config_path: str = "config/models.yaml"):
        self.hub = ModelHub(config_path)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def distill(self, teacher_name: str, student_name: str,
                prompts: list, output_dir: str, num_epochs: int = 3) -> str:
        """Generate teacher answers then fine-tune the student on them."""
        print(f"\n🎓 Distillation  teacher={teacher_name}  student={student_name}  epochs={num_epochs}")
        print("-" * 55)

        # 1. Collect teacher responses
        print("📝 Generating teacher data …")
        texts = []
        for i, p in enumerate(prompts):
            resp = self.hub.generate_response(teacher_name, p, max_tokens=100)
            texts.append(f"Question: {p}\nAnswer: {resp}")
            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                print(f"  {i+1}/{len(prompts)}")

        # 2. Load student
        info = self.hub.get_model_info(student_name)
        if info["provider"] == "ollama":
            raise ValueError("Cannot distill into Ollama models – use a HuggingFace model")

        model = AutoModelForCausalLM.from_pretrained(student_name)
        tok = AutoTokenizer.from_pretrained(student_name)
        tok.pad_token = tok.eos_token

        # 3. Tokenise
        max_len = self.config["training"].get("max_length", 128)
        ds = Dataset.from_dict({"text": texts})
        ds = ds.map(lambda ex: tok(ex["text"], truncation=True, padding="max_length", max_length=max_len),
                     batched=True)

        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config["training"].get("batch_size", 2),
            save_strategy="epoch",
            logging_steps=5,
            learning_rate=float(self.config["training"].get("learning_rate", 5e-5)),
            report_to="none",
        )

        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)

        print("🔥 Training …")
        trainer.train()

        final = os.path.join(output_dir, "final")
        model.save_pretrained(final)
        tok.save_pretrained(final)
        print(f"✅ Saved → {final}")
        return final
