# Stage 04 SLM: Cloud GPU Training & Evaluation Guide

**Personalized Precision Medicine for Oncology Treatment Optimization**  
**Model:** `Qwen/Qwen2.5-3B` + PEFT/LoRA ($r=16, \alpha=32, \text{dropout}=0.05$)  
**Goal:** Supervised Fine-Tuning (SFT) for faithful clinical summarization connecting Stage 03 NLP to Stage 06 Multi-Agent System.

> [!IMPORTANT]
> **No Compilation Required:**
> - `flash-attn` is **strictly not required**. Standard PyTorch native Scaled Dot-Product Attention (`sdpa`) / eager attention is used.
> - `torchaudio` is **strictly not required**.
> - Training runs out-of-the-box on standard NVIDIA GPUs (T4 16GB, L4 24GB, A100 40/80GB, H100, RTX 3090/4090).

---

## 1. Creating a GPU Environment

Choose any cloud provider or GPU instance:
- **Google Colab:** Select **Runtime > Change runtime type > T4 GPU** or **A100 GPU**.
- **Kaggle Notebooks:** Select **Settings > Accelerator > GPU T4 x 2** or **P100**.
- **Lambda Labs / RunPod / Vast.ai:** Launch an instance with Ubuntu 22.04 and 1x NVIDIA GPU (RTX 3090/4090, A10, L4, or A100).
- **AWS / Azure:** EC2 `g5.xlarge` (A10G 24GB) or Azure `Standard_NC4as_T4_v3`.

Verify that the GPU and CUDA driver are accessible:
```bash
nvidia-smi
```

---

## 2. Installing Compatible PyTorch + Transformers + PEFT

Create an isolated virtual environment and install the required dependencies using [`requirements_gpu.txt`](file:///c:/Users/rethi/OneDrive/%E0%B9%80%E0%B8%AD%E0%B8%81%E0%B8%AA%E0%B8%B2%E0%B8%A3/Desktop/DS%20team%20pro/requirements_gpu.txt):

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 support (WITHOUT torchaudio)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install Transformers, PEFT, Accelerate, Datasets, and Evaluation libraries
pip install -r stage4_slm/requirements_gpu.txt
```

Verify GPU detection in Python:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Downloading Pretrained `Qwen/Qwen2.5-3B`

The training and evaluation scripts will automatically download the base model and tokenizer from Hugging Face Hub upon first run. 

Alternatively, pre-cache the model in the background:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B'); \
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', torch_dtype='auto', low_cpu_mem_usage=True); \
print('Qwen2.5-3B downloaded successfully.')"
```

*(Optional: If your environment requires Hugging Face authentication to raise rate limits, run `huggingface-cli login` or export `HF_TOKEN`)*.

---

## 4. Running LoRA Training

Execute the SFT training pipeline. The script automatically:
- Detects the GPU and allocates memory efficiently.
- Uses standard PyTorch SDPA attention (no flash-attn compilation).
- Applies target-only loss masking (calculates CrossEntropy loss solely on assistant summary tokens).
- Excludes `Patient_ID` from model inputs.
- Evaluates on `validation.jsonl` at each epoch and saves the best checkpoint.
- Keeps `test.jsonl` strictly untouched.

```bash
python stage4_slm/train_slm.py \
    --model_name "Qwen/Qwen2.5-3B" \
    --train_file "stage4_slm/data/train.jsonl" \
    --val_file "stage4_slm/data/validation.jsonl" \
    --test_file "stage4_slm/data/test.jsonl" \
    --output_dir "stage4_slm/models/qwen2.5_3b_lora" \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 2e-4 \
    --max_length 256 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

### Memory Optimization for Different GPUs:
- **For 16GB GPUs (T4 / V100):** Keep `--batch_size 2 --gradient_accumulation 16 --gradient_checkpointing`.
- **For 24GB GPUs (RTX 3090 / 4090 / L4 / A10G):** Use `--batch_size 4 --gradient_accumulation 8`.
- **For 40GB/80GB GPUs (A100 / H100):** Use `--batch_size 8 --gradient_accumulation 4`.

---

## 5. Running Final Test Evaluation

Evaluate the fine-tuned LoRA adapter strictly on the **447 untouched holdout test records** (`stage4_slm/data/test.jsonl`):

```bash
python stage4_slm/evaluate_slm.py \
    --base_model "Qwen/Qwen2.5-3B" \
    --adapter_dir "stage4_slm/models/qwen2.5_3b_lora" \
    --test_file "stage4_slm/data/test.jsonl" \
    --reports_dir "stage4_slm/reports" \
    --outputs_dir "stage4_slm/outputs" \
    --max_new_tokens 128
```

This generates:
- `stage4_slm/outputs/predictions.csv`: Predictions across all 447 test samples.
- `stage4_slm/outputs/sample_predictions.json`: Detailed prompt-target-generated outputs.
- `stage4_slm/reports/evaluation_report.md`: Formal benchmark report containing ROUGE-1/2/L, entity preservation rates (drugs, mutations, dosages, adverse events), sequence length stats, and 6 clinical case studies.
- `stage4_slm/reports/model_comparison.csv`: Summary metrics table.

---

## 6. Saving and Packaging the LoRA Adapter

The LoRA adapter is saved separately from the 5.75 GB base model, producing a lightweight, portable artifact package (~115 MB):

```
stage4_slm/models/qwen2.5_3b_lora/
├── adapter_model.safetensors    # Trainable LoRA weights (29.93M parameters)
├── adapter_config.json          # PEFT configuration (r=16, alpha=32, target modules)
├── tokenizer.json               # Qwen2.5 tokenizer
├── tokenizer_config.json        # Special tokens (<|im_start|>, <|im_end|>)
└── special_tokens_map.json
```

To compress the adapter package for download:
```bash
tar -czvf qwen2.5_3b_lora_weights.tar.gz stage4_slm/models/qwen2.5_3b_lora/
```

You can then download `qwen2.5_3b_lora_weights.tar.gz` back to your local environment.

---

## 7. Running Inference

### CLI Inference
Run deterministic clinical summarization with integrated safety guardrails:
```bash
python stage4_slm/inference.py \
    --base_model "Qwen/Qwen2.5-3B" \
    --adapter "stage4_slm/models/qwen2.5_3b_lora" \
    --text "Nurse intake: Pt diagnosed with EGFR L858R NSCLC. Started Erlotinib 150 mg daily. Complains of mild nausea after intake, no severe toxicities reported."
```

Or pass a text file containing an intake note:
```bash
python stage4_slm/inference.py --file "path/to/clinical_note.txt"
```

### Production REST API Endpoint
Launch the FastAPI server:
```bash
python -m uvicorn stage4_slm.api:app --host 0.0.0.0 --port 8000
```

Query the `/summarize` endpoint:
```bash
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{
       "clinical_report": "Patient with BRAF V600E metastatic melanoma initiated Dabrafenib 150 mg BID. Reported mild pyrexia.",
       "urgency": "Moderate",
       "entities": {"drugs": ["Dabrafenib"], "mutations": ["BRAF V600E"]}
     }'
```
