# [Paste the full DistillationTrainer code from earlier - I'll provide a concise but functional version]
import os, yaml, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from core.model_hub import ModelHub
from core.observability import get_observability

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    PEFT_AVAILABLE = True
except: PEFT_AVAILABLE = False

class DistillationTrainer:
    def __init__(self, config_path="config/models.yaml"):
        self.hub = ModelHub(config_path)
        with open(config_path) as f: self.config = yaml.safe_load(f)
        self.obs = get_observability()
    def distill(self, teacher_name, student_name, prompts, output_dir, num_epochs=3, use_peft=True, use_qlora=True, lora_r=16, lora_alpha=32):
        self.obs.trace_distillation(teacher_name, student_name, len(prompts), output_dir)
        print(f"\n🎓 Distillation: {teacher_name} → {student_name}")
        texts = [f"### Instruction:\n{p}\n\n### Response:\n{self.hub.generate_response(teacher_name, p, max_tokens=200)}" for p in prompts]
        info = self.hub.get_model_info(student_name)
        if info["provider"] == "ollama": raise ValueError("Cannot distill into Ollama")
        model_kwargs = {}
        if use_peft and use_qlora and PEFT_AVAILABLE:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(student_name, device_map="auto", **model_kwargs)
        tok = AutoTokenizer.from_pretrained(student_name); tok.pad_token = tok.eos_token
        ds = Dataset.from_dict({"text": texts})
        if use_peft and PEFT_AVAILABLE:
            if use_qlora: model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
            model = get_peft_model(model, peft_config)
        training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=num_epochs, per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=10, logging_steps=5, save_strategy="epoch", learning_rate=2e-4 if use_peft else 5e-5, fp16=torch.cuda.is_available(), report_to="none", remove_unused_columns=False)
        if PEFT_AVAILABLE and use_peft:
            trainer = SFTTrainer(model=model, args=training_args, train_dataset=ds, tokenizer=tok, max_seq_length=self.config["training"].get("max_length",512), dataset_text_field="text")
        else:
            from transformers import Trainer, DataCollatorForLanguageModeling
            def tokenize(ex): return tok(ex["text"], truncation=True, padding="max_length", max_length=512)
            ds = ds.map(tokenize, batched=True)
            trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer.train()
        final = os.path.join(output_dir, "final")
        if use_peft and PEFT_AVAILABLE: model.save_pretrained(final)
        else: model.save_pretrained(final)
        tok.save_pretrained(final)
        print(f"✅ Saved → {final}")
        return final
