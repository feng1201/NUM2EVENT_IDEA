# Num2Event — Experiment Pipeline

---

## 1) Environment setup

### Option A: Conda

```bash
conda create -n num2event python=3.11 -y
conda activate num2event
pip install -r requirements.txt
```

---


## 2) Model & API Configuration

### Download Model Weights
We recommend using the `huggingface-cli` to download the model efficiently.

```bash
# 1. Install Hugging Face Hub CLI
pip install -U "huggingface_hub[cli]"

# 2. Download the model to a local directory
huggingface-cli download Qwen/Qwen3-8B --local-dir checkpoints/Qwen3-8B
```
### API Configuration

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_BASE_URL"
```

## 3) Event extraction & Schema expansion

This step reads `dataset/energy.csv` and extracts structured events from the text column, writing outputs under `eventsdata_en/`.

```bash
python extract/extract_energy_events_simple_date_en.py \
  --input dataset/energy.csv \
  --model gpt-4o-mini \
  --max-concurrency 100
```

Expected outputs (written automatically):
- `eventsdata_en/events_raw.jsonl`
- `eventsdata_en/events_flat_simple.jsonl` (the downstream pipeline input)

---


During extraction, the pipeline also writes per-row vocabulary suggestions to:
- `eventsdata_en/vocab_suggestions.jsonl`

You can aggregate these suggestions and keep only high-support terms (by count) to expand your schema:

```bash
python extract/discover_vocab_bootstrap.py \
  --input eventsdata_en/vocab_suggestions.jsonl \
  --out eventsdata_en/vocab_suggestions_filtered.jsonl \
  --thr 10
```

Key arguments:
- `--input`: input JSONL (default: `eventsdata_en/vocab_suggestions.jsonl`)
- `--out`: output JSONL with `term/slot/count/avg_score` (default: `eventsdata_en/vocab_suggestions_filtered.jsonl`)
- `--thr`: minimum **count** threshold (default: `10`)
- `--min-score`: optional minimum **average score** threshold (default: `0.0`)
- `--sort`: sort key, `count` or `avg_score` (default: `count`)

---

## 4) Split events by date

This produces the deduplicated event sets used for training/validation/testing.

```bash
python data_process/energy4experiment_pipeline_standalone.py \
  --input-events eventsdata_en/events_flat_simple.jsonl
```

Key outputs (defaults):
- `data4dedup_en/events_dedup_min5topk1000_before2022_train.jsonl`
- `data4dedup_en/events_dedup_min5topk1000_during2022_2023_vail.jsonl`
- `data4dedup_en/events_dedup_min5topk1000_after2023_test.jsonl`
- `data4dedup_en/events_dedup_min1topk1000_after2023_test_stat.jsonl`

Tip:

```bash
python data_process/energy4experiment_pipeline_standalone.py --dry-run
```

---

## 5) Synthesis (IRF + arrivals model)

This generates synthetic data under a chosen output directory.

```bash
python syn/synthesis_en.py \
  --energy_csv dataset/energy.csv \
  --events_jsonl data4dedup_en/events_dedup_min5topk1000_before2022_train.jsonl \
  --outdir syndata_en/syndata_v1 \
  --min_count 2 \
  --irf_h 8 \
  --spillover_mode none \
  --max_categories 30 \
  --irf_clip 0.3 \
  --no_energy_filter \
  --arrivals_model hawkes \
  --hawkes_l1 0.005 \
  --hawkes_rho_bounds 0.3,0.95 \
  --hawkes_cap_mult 10 \
  --make_sft_rl \
  --lookback_months 12 \
  --gen_months 3 \
  --lambda_scale 0.1 \
  --rng_seed 333 \
  --monthly_stride 1 \
  --event_randomness 1.0
```

---

## 6) Build SFT datasets + reasoning traces

This step builds the SFT JSONL files (real + synthetic), optionally generates reasoning traces (Reason + FINAL), merges datasets, and rewrites prompts.

```bash
python data_process/energy4experiment_sft_full_standalone.py \
  --reason-model gpt-4o-mini
```

If you want to control where the merged train + synthetic JSONL comes from:

```bash
python data_process/energy4experiment_sft_full_standalone.py \
  --merge-train-jsonl path/to/train.jsonl \
  --merge-syn-jsonl path/to/syn.jsonl
```

---

## 7) Training

### Stage 1 (TS Encoder)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=8 finetune/finetune_twostage.py \
  --base_model weight/Qwen38b \
  --train_jsonl data/energy_train.jsonl \
  --valid_jsonl data/energy_vail.jsonl \
  --out_dir Qwen_weight_en/syn_stage1_patch2_128d_4bs_10epochs \
  --stage 1 \
  --bf16 --epochs 100 --lr 1e-4 \
  --batch_size 4 \
  --grad_accum 4 --patch_size_ot 2 --patch_size_dot 2 --ts_hidden 128 --ts_layers 4 \
  --gradient_checkpointing \
  --logging_steps 10 --eval_steps 200
```

### Stage 2 (Cold Start)

```bash
export PYTHONUNBUFFERED=1
export MASTER_PORT=29511

torchrun --nproc_per_node=8 --master_port=${MASTER_PORT} finetune/finetune_twostage.py \
  --base_model weight/Qwen38b \
  --train_jsonl data/energy_train.jsonl \
  --valid_jsonl data/energy_vail.jsonl \
  --out_dir Qwen_weight_en/syn_stage2_patch2_128d_4bs_10epochs \
  --stage1_dir Qwen_weight_en/syn_stage1_patch2_128d_4bs_10epochs/epoch_ckpts/epoch_10 \
  --stage 2 \
  --bf16 --epochs 1 --lr 1e-6 \
  --batch_size 1 \
  --grad_accum 4 --patch_size_ot 2 --patch_size_dot 2 --ts_hidden 128 --ts_layers 4 \
  --gradient_checkpointing \
  --logging_steps 10 --eval_steps 200
```

### Stage 3 (GRPO)

If your GRPO code depends on `trl`’s `grpo_trainer.py`, you may need to patch or redirect imports to use the local implementation.

```bash
export MASTER_PORT=29520
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=8 --master_port=${MASTER_PORT} finetune/grpo_stage.py \
  --base_model weight/Qwen38b \
  --stage1_dir Qwen_weight_en/syn_stage1_patch2_128d_4bs_10epochs/epoch_ckpts/epoch_10 \
  --train_jsonl data/energy_train.jsonl \
  --sft_lora_dir Qwen_weight_en/syn_stage2_patch2_128d_4bs_10epochs/epoch_ckpts/epoch_01 \
  --out_dir Qwen_weight_en/syn_stage3_patch2_128d_4bs_10epochs \
  --bf16 \
  --use_lora
```

---

## 8) Evaluation

```bash
python eval_en_RL_new_server_score_log/run_eval_ts.py \
  --data data/energy_test.jsonl \
  --base weight/Qwen38b \
  --adapter Qwen_weight_en/syn_stage3_patch2_128d_4bs_10epochs/epoch_ckpts/epoch_01 \
  --out result_energy_csv/ours_log.csv \
  --log_dir result_energy_log/ours_energy.log\
  --ts-ckpt Qwen_weight_en/syn_stage1_patch2_128d_4bs_10epochs/epoch_ckpts/epoch_10
```


