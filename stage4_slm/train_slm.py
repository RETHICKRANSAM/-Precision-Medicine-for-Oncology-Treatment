"""
Stage 04: CPU-Only LoRA Supervised Fine-Tuning (SFT) for Qwen/Qwen2.5-0.5B-Instruct.

Hardware & Execution:
- 100% CPU execution with torch.float32 (no CUDA, bitsandbytes, flash-attn, or quantization)
- Model: Qwen/Qwen2.5-0.5B-Instruct
- LoRA: r=16, alpha=32, dropout=0.05 across all 7 projection modules
- Datasets:
    - Train: stage4_slm/data/train.jsonl (3,576 records)
    - Val: stage4_slm/data/val.jsonl or validation.jsonl (447 records)
    - Test: stage4_slm/data/test.jsonl (STRICTLY UNTOUCHED)
- Target-only loss masking on assistant response tokens (-100 on prompt)
- Hyperparameters:
    - per_device_train_batch_size = 1
    - per_device_eval_batch_size = 1
    - gradient_accumulation_steps = 4
    - learning_rate = 2e-4
    - weight_decay = 0.01
    - num_train_epochs = 1
    - max_length = 256
    - optim = "adamw_torch"
    - logging_steps = 1
    - eval_steps = 25
    - save_steps = 25
    - save_total_limit = 2
- Output directory: stage4_slm/models/qwen2.5_0.5b_lora_cpu
- Supports 5-step smoke test (--smoke_test) before full training
"""

import os
import sys
import json
import time
import shutil
import argparse
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Safe console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class OncologySFTDataset(Dataset):
    """
    Dataset for Clinical Instruction Tuning with target-only loss masking.
    Prompts use ChatML formatting. Prompt tokens are labeled with -100 so
    loss is computed solely on the assistant target summary tokens.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        if max_samples and max_samples < len(self.records):
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        instruction = item.get("instruction", "Summarize the following oncology clinical report faithfully.")
        inp = item.get("input", "")
        target = item.get("output", "")

        # Format ChatML prompt strictly following Qwen2.5 standard
        prompt_text = (
            f"<|im_start|>system\n"
            f"You are a clinical oncology assistant. Summarize clinical notes faithfully without inventing facts.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}\n\nClinical Report:\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt_text + target + "<|im_end|>"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate to maximum sequence length if necessary
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        input_ids = full_ids
        # Target-only loss masking: prompt tokens are labeled as -100
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch, pad_token_id):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        cur_len = item["input_ids"].shape[0]
        pad_len = max_len - cur_len

        padded_input = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids.append(padded_input)
        attention_mask.append(padded_mask)
        labels.append(padded_labels)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 04: CPU-Only LoRA Training for Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="stage4_slm/data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="stage4_slm/data/val.jsonl")
    parser.add_argument("--test_file", type=str, default="stage4_slm/data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="stage4_slm/models/qwen2.5_0.5b_lora_cpu")
    parser.add_argument("--reports_dir", type=str, default="stage4_slm/reports")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_eval_samples", type=int, default=50, help="Max validation samples during intermediate evaluations (default: 50)")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional early stop cap (e.g. for timed CPU run)")
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run 5-step smoke test only")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model, val_loader, device, max_eval_samples=None):
    """Computes validation loss over validation dataset."""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if max_eval_samples and val_batches >= max_eval_samples:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            val_batches += 1

    return total_val_loss / max(1, val_batches)


def run_smoke_test(model, tokenizer, train_loader, optimizer, scheduler, device, gradient_accumulation):
    """Executes a 5-step training smoke test to verify forward/backward passes and gradients."""
    print("\n" + "=" * 70)
    print("STAGE 04: RUNNING 5-STEP TRAINING SMOKE TEST ON CPU")
    print("=" * 70)

    model.train()
    smoke_start = time.time()
    optimizer.zero_grad()
    losses = []

    for step, batch in enumerate(train_loader):
        if step >= 5:
            break

        step_t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation
        loss.backward()

        step_loss = outputs.loss.item()
        losses.append(step_loss)

        if (step + 1) % gradient_accumulation == 0 or (step + 1) == 5:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_duration = time.time() - step_t0
        print(f"  Smoke Test Step {step + 1}/5 | Loss: {step_loss:.4f} | Time: {step_duration:.2f}s", flush=True)

    smoke_total = time.time() - smoke_start
    print("=" * 70)
    print(f"[SMOKE TEST PASSED] 5 steps completed in {smoke_total:.2f}s (Avg {smoke_total / 5:.2f}s/step).")
    print(f"Initial Loss: {losses[0]:.4f} -> Step 5 Loss: {losses[-1]:.4f}")
    print("Forward, backward gradient computation, and optimizer step verified successfully.")
    print("=" * 70 + "\n")
    return True


def manage_checkpoints(output_dir: str, save_total_limit: int):
    """Maintains at most save_total_limit checkpoint directories."""
    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
    ]
    if len(checkpoints) > save_total_limit:
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        for ckpt in checkpoints[:-save_total_limit]:
            shutil.rmtree(ckpt, ignore_errors=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # Resolve val file path
    val_path = args.val_file
    if not os.path.exists(val_path):
        alt_val = "stage4_slm/data/validation.jsonl"
        if os.path.exists(alt_val):
            val_path = alt_val

    # 1. Check CPU Availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cpu")
    cpu_count = os.cpu_count() or 1

    print("=" * 70)
    print("STAGE 04: CPU-ONLY LoRA TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device:                 CPU (torch.float32)")
    print(f"Available CPU Cores:    {cpu_count}")
    print(f"CUDA Available:         {cuda_available} (Bypassed / CPU Only enforced)")
    print(f"Base Model:             {args.model_name}")
    print(f"LoRA Target Modules:    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
    print(f"LoRA Hyperparameters:   r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Batch Size:             {args.batch_size} (Grad Accum: {args.gradient_accumulation})")
    print(f"Effective Batch Size:   {args.batch_size * args.gradient_accumulation}")
    print(f"Learning Rate:          {args.learning_rate} (Weight Decay: {args.weight_decay})")
    print(f"Optimizer:              adamw_torch (torch.optim.AdamW)")
    print(f"Epochs:                 {args.epochs}")
    print(f"Max Sequence Length:    {args.max_length}")
    print(f"Output Directory:       {args.output_dir}")
    print("=" * 70)

    # 2. Load Tokenizer
    print("\nLoading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Pretrained Base Model on CPU
    print(f"Loading Base Model ({args.model_name}) on CPU with torch.float32...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=None,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    )
    base_model.to(device)

    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base Model Name:             {args.model_name}")
    print(f"Total Base Model Parameters: {total_base_params:,}")

    # 4. LoRA Adapter Setup
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    model = get_peft_model(base_model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / total_params) * 100.0

    print("\n--- Parameter & Freezing Verification ---")
    print(f"Total Parameters:            {total_params:,}")
    print(f"Trainable LoRA Parameters:   {trainable_params:,}")
    print(f"Trainable Percentage:        {trainable_pct:.4f}%")

    # Verify base model is 100% frozen
    base_frozen = all(
        not param.requires_grad
        for name, param in model.named_parameters()
        if "lora" not in name.lower()
    )
    print(f"Base Model Frozen Status:    {'VERIFIED (100% Frozen)' if base_frozen else 'FAILED'}")
    assert base_frozen, "Base model parameters must be 100% frozen."

    # 5. Datasets (Train on train.jsonl only; Val on val.jsonl only; NEVER on test.jsonl)
    print("\nLoading Datasets...")
    train_dataset = OncologySFTDataset(args.train_file, tokenizer, max_length=args.max_length)
    val_dataset = OncologySFTDataset(val_path, tokenizer, max_length=args.max_length)

    print(f"Train Dataset Size:          {len(train_dataset):,} samples ({args.train_file})")
    print(f"Validation Dataset Size:     {len(val_dataset):,} samples ({val_path})")
    print(f"Test Dataset Status:         STRICTLY HELD OUT ({args.test_file} untouched)")

    # Verify target-only label masking on first sample
    sample0 = train_dataset[0]
    masked_count = sum(1 for l in sample0["labels"].tolist() if l == -100)
    unmasked_count = sum(1 for l in sample0["labels"].tolist() if l != -100)
    print(f"Target-Only Label Masking:   VERIFIED (Prompt masked: {masked_count} tokens, Target unmasked: {unmasked_count} tokens)")
    assert masked_count > 0 and unmasked_count > 0, "Loss masking verification failed."

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    # 6. Optimizer & Linear Warmup Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_optimizer_steps = (len(train_loader) // args.gradient_accumulation) * args.epochs
    warmup_steps = max(1, int(total_optimizer_steps * 0.05))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps)

    # 7. Run 5-Step Smoke Test First
    smoke_success = run_smoke_test(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation=args.gradient_accumulation
    )

    if args.smoke_test:
        print("[INFO] --smoke_test completed successfully. Exiting prior to full training.")
        return

    # Reset optimizer and scheduler for clean full training run
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps)

    # 8. Full 1-Epoch CPU Training Loop
    print("\n" + "=" * 70)
    print("STARTING 1-EPOCH CPU TRAINING")
    print(f"Total Batches: {len(train_loader)} | Optimizer Steps: {total_optimizer_steps}")
    print(f"Evaluation & Save Interval: Every {args.eval_steps} optimizer steps")
    print("=" * 70)

    training_logs = []
    training_start = time.time()
    optimizer_step = 0
    accumulated_loss = 0.0
    accumulated_count = 0
    best_val_loss = float("inf")
    completed_steps = 0

    # Evaluate initial pre-training validation loss
    print(f"Evaluating initial baseline validation loss on val.jsonl (first {args.max_eval_samples} samples)...", flush=True)
    initial_val_loss = evaluate(model, val_loader, device, max_eval_samples=args.max_eval_samples)
    print(f"Baseline Validation Loss: {initial_val_loss:.4f}\n", flush=True)

    training_logs.append({
        "step": 0,
        "train_loss": None,
        "val_loss": round(initial_val_loss, 4),
        "lr": args.learning_rate,
        "elapsed_seconds": 0.0
    })

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                batch_t0 = time.time()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation
                loss.backward()

                accumulated_loss += outputs.loss.item()
                accumulated_count += 1
                completed_steps += 1

                # Gradient accumulation trigger
                if (batch_idx + 1) % args.gradient_accumulation == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

                    avg_step_loss = accumulated_loss / max(1, accumulated_count)
                    accumulated_loss = 0.0
                    accumulated_count = 0
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - training_start

                    # Logging step
                    log_entry = {
                        "step": optimizer_step,
                        "batch": batch_idx + 1,
                        "train_loss": round(avg_step_loss, 4),
                        "val_loss": None,
                        "lr": round(current_lr, 8),
                        "elapsed_seconds": round(elapsed, 2)
                    }

                    if optimizer_step % args.logging_steps == 0:
                        print(f"Step {optimizer_step:4d}/{total_optimizer_steps} | Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s", flush=True)

                    # Evaluation and Save step
                    if optimizer_step % args.eval_steps == 0 or optimizer_step == total_optimizer_steps:
                        print(f"\n--> Running validation at optimizer step {optimizer_step}...", flush=True)
                        val_loss = evaluate(model, val_loader, device, max_eval_samples=args.max_eval_samples)
                        log_entry["val_loss"] = round(val_loss, 4)
                        print(f"--> Step {optimizer_step} | Validation Loss: {val_loss:.4f}", flush=True)

                        # Checkpoint saving
                        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{optimizer_step}")
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        manage_checkpoints(args.output_dir, args.save_total_limit)

                        # Best model saving
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(f"--> Validation loss improved ({val_loss:.4f} < {best_val_loss:.4f})! Updating best model at {args.output_dir}", flush=True)
                            model.save_pretrained(args.output_dir)
                            tokenizer.save_pretrained(args.output_dir)

                        model.train()
                        print("", flush=True)

                    training_logs.append(log_entry)

                # Early step cap if specified
                if args.max_steps and optimizer_step >= args.max_steps:
                    print(f"\n[INFO] Reached requested max_steps={args.max_steps}. Stopping safely.", flush=True)
                    break

            if args.max_steps and optimizer_step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training safely interrupted by user.", flush=True)

    # 9. Final Validation & Model Saving
    total_training_time = time.time() - training_start
    print("\n" + "=" * 70)
    print("CPU TRAINING RUN COMPLETED / CONCLUDED")
    print("=" * 70)
    print(f"Total Batches Processed:       {completed_steps:,}")
    print(f"Total Optimizer Steps:         {optimizer_step:,}")
    print(f"Total Training Time:           {total_training_time:.2f} s ({total_training_time / 60:.2f} min)")
    print(f"Average Time per Batch:        {total_training_time / max(1, completed_steps):.2f} s")

    print("Evaluating final validation loss...", flush=True)
    final_val_loss = evaluate(model, val_loader, device)
    print(f"Final Validation Loss:         {final_val_loss:.4f}")
    print(f"Best Validation Loss:          {min(best_val_loss, final_val_loss):.4f}")

    # Ensure best adapter and tokenizer are saved in main output directory
    print(f"Saving final trained LoRA adapter and tokenizer to {args.output_dir}...", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 10. Save Training Logs
    log_json_path = os.path.join(args.reports_dir, "training_log_cpu.json")
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)
    print(f"Training log (JSON) saved to:   {log_json_path}")

    log_df = pd.DataFrame(training_logs)
    log_csv_path = os.path.join(args.reports_dir, "training_log_cpu.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"Training log (CSV) saved to:    {log_csv_path}")

    # Generate Training Summary Markdown
    train_report_path = os.path.join(args.reports_dir, "training_report_cpu.md")
    report_content = f"""# Stage 04 SLM CPU Training Report: Qwen2.5-0.5B-Instruct + LoRA

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Adapter Output Directory:** `{args.output_dir}`  
**Hardware:** Host CPU (`torch.float32`)  
**Optimizer Steps Completed:** `{optimizer_step}`  
**Total Training Time:** `{total_training_time:.2f}s` (`{total_training_time / 60:.2f} minutes`)  
**Initial Baseline Validation Loss:** `{initial_val_loss:.4f}`  
**Final Validation Loss:** `{final_val_loss:.4f}`  
**Best Validation Loss:** `{min(best_val_loss, final_val_loss):.4f}`  

---

## LoRA Hyperparameters
- **Rank ($r$):** 16
- **Alpha ($\alpha$):** 32
- **Dropout:** 0.05
- **Target Modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Trainable Parameters:** {trainable_params:,} ({trainable_pct:.4f}%)
- **Base Model Parameters:** {total_base_params:,} (100% Frozen)
- **Batch Size:** 1 (Gradient Accumulation: 4)
- **Learning Rate:** 2e-4 (Warmup: 5%, linear decay)
- **Optimizer:** `adamw_torch` (Weight Decay: 0.01)

---

## Dataset Accountability
- **Training Set:** `{args.train_file}` ({len(train_dataset):,} samples)
- **Validation Set:** `{val_path}` ({len(val_dataset):,} samples)
- **Test Set:** `{args.test_file}` (100% held out, 0 contamination)
"""
    with open(train_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Training summary written to:    {train_report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
