"""Trainer v3 — PEFT/LoRA distillation for 4GB VRAM GPUs.

Critical fix: auto-detects LoRA target modules per architecture.
  - GPT-2/DistilGPT-2: c_attn, c_proj
  - Qwen/Llama/Mistral/Gemma: q_proj, k_proj, v_proj, o_proj, etc.

Falls back to full fine-tuning if PEFT is not installed.
"""

import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from core.model_hub import ModelHub

try:
    from transformers import BitsAndBytesConfig
    BNB_OK = True
except ImportError:
    BNB_OK = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_OK = True
except ImportError:
    PEFT_OK = False

try:
    from trl import SFTTrainer
    TRL_OK = True
except ImportError:
    TRL_OK = False

# Known LoRA target modules per model architecture
LORA_TARGETS = {
    "gpt2":    ["c_attn", "c_proj"],
    "gpt_neo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "llama":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi":     ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
}


def detect_lora_targets(model):
    """Auto-detect which LoRA target modules exist in the model."""
    model_type = getattr(model.config, "model_type", "").lower()
    candidates = LORA_TARGETS.get(model_type, [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "c_attn", "c_proj", "c_fc",
        "gate_proj", "up_proj", "down_proj",
    ])
    all_names = set()
    for name, _ in model.named_modules():
        for part in name.split("."):
            all_names.add(part)
    valid = [m for m in candidates if m in all_names]
    if not valid:
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                short = name.split(".")[-1]
                if short not in valid:
                    valid.append(short)
                if len(valid) >= 4:
                    break
    print(f"    LoRA targets ({model_type}): {valid}")
    return valid


class DistillationTrainer:
    """Knowledge distillation with PEFT/LoRA for low-VRAM GPUs."""

    def __init__(self, config_path="config/models.yaml"):
        self.hub = ModelHub(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def distill(self, teacher_name, student_name, training_texts,
                output_dir, num_epochs=3, use_peft=True,
                use_qlora=True, lora_r=16, lora_alpha=32):
        method = "QLoRA" if use_qlora and PEFT_OK and BNB_OK else \
                 "LoRA" if use_peft and PEFT_OK else "Full"
        print(f"\n  🎓 Distillation: {teacher_name.split('/')[-1]} → {student_name.split('/')[-1]}")
        print(f"     Method: {method} | Data: {len(training_texts)} examples | Epochs: {num_epochs}")

        info = self.hub.get_model_info(student_name)
        if info["provider"] == "ollama":
            raise ValueError("Cannot distill into Ollama models")
        if not training_texts:
            print("  ⚠️ No training data — skipping")
            return output_dir

        # Format training data
        texts = []
        for t in training_texts:
            if "Answer:" in t or "Response:" in t:
                texts.append(f"### Instruction:\n{t}")
            else:
                try:
                    resp = self.hub.generate_response(teacher_name, t, max_tokens=200)
                    texts.append(f"### Instruction:\n{t}\n\n### Response:\n{resp}")
                except Exception:
                    texts.append(f"### Instruction:\n{t}\n\n### Response:\n(no response)")

        # Load student model
        model_kwargs = {}
        if method == "QLoRA":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            student_name, device_map="auto", **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(student_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = Dataset.from_dict({"text": texts})

        # Apply LoRA with auto-detected targets
        if method in ("LoRA", "QLoRA") and PEFT_OK:
            if method == "QLoRA":
                model = prepare_model_for_kbit_training(model)
            targets = detect_lora_targets(model)
            if not targets:
                print("  ⚠️ No LoRA targets found — falling back to full training")
                method = "Full"
            else:
                peft_config = LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                    bias="none", task_type="CAUSAL_LM", target_modules=targets)
                model = get_peft_model(model, peft_config)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"     Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

        training_args = TrainingArguments(
            output_dir=output_dir, num_train_epochs=num_epochs,
            per_device_train_batch_size=1 if method == "Full" else 2,
            gradient_accumulation_steps=8 if method == "Full" else 4,
            warmup_steps=10, logging_steps=5, save_strategy="epoch",
            learning_rate=2e-4 if method != "Full" else 5e-5,
            fp16=torch.cuda.is_available(), report_to="none",
            remove_unused_columns=False)

        if method in ("LoRA", "QLoRA") and TRL_OK:
            trainer = SFTTrainer(
                model=model, args=training_args, train_dataset=ds,
                processing_class=tokenizer,
                max_seq_length=self.config["training"].get("max_length", 512),
                dataset_text_field="text")
        else:
            from transformers import Trainer, DataCollatorForLanguageModeling
            def tok_fn(ex):
                return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)
            ds = ds.map(tok_fn, batched=True)
            trainer = Trainer(model=model, args=training_args, train_dataset=ds,
                              data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

        trainer.train()
        final_path = os.path.join(output_dir, "final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"     ✅ Saved → {final_path}")
        return final_path
