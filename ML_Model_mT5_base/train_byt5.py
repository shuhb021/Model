import os
import pandas as pd
import numpy as np
import sacrebleu
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerControl,
    TrainerState
)

# ── WINDOWS DEADLOCK PREVENTION ──────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ── 1. LOAD & SPLIT DATA ──────────────────────────────────────────────────────
csv_path = "GONDI_Dictionary_Plain.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Gondi', 'English', 'Hindi'])

# Split BEFORE pair expansion to prevent data leakage
train_df = df.sample(frac=0.8, random_state=42)
temp_df  = df.drop(train_df.index)
valid_df = temp_df.sample(frac=0.5, random_state=42)
test_df  = temp_df.drop(valid_df.index)

# ── 2. CREATE 6-WAY TRANSLATION PAIRS ────────────────────────────────────────
def create_pairs(dataframe):
    pairs = []
    for _, row in dataframe.iterrows():
        gondi   = str(row['Gondi']).strip()
        english = str(row['English']).strip()
        hindi   = str(row['Hindi']).strip()
        if not gondi or not english or not hindi:
            continue
        directions = [
            (f"Translate English to Gondi: {english}", gondi),
            (f"Translate Gondi to English: {gondi}",   english),
            (f"Translate Hindi to Gondi: {hindi}",     gondi),
            (f"Translate Gondi to Hindi: {gondi}",     hindi),
            (f"Translate English to Hindi: {english}", hindi),
            (f"Translate Hindi to English: {hindi}",   english),
        ]
        pairs.extend([{"source": src, "target": tgt} for src, tgt in directions])
    return Dataset.from_list(pairs)

datasets = DatasetDict({
    'train':      create_pairs(train_df),
    'validation': create_pairs(valid_df),
    'test':       create_pairs(test_df),
})
print(f"Train: {len(datasets['train'])} | Val: {len(datasets['validation'])} | Test: {len(datasets['test'])}")

# ── 3. TOKENISE ───────────────────────────────────────────────────────────────
model_checkpoint = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs  = tokenizer(examples["source"], max_length=64, truncation=True)
    targets = tokenizer(examples["target"], max_length=64, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

# No num_proc — this datasets version spawns a pool even for num_proc=1,
# which crashes on Windows when called outside __main__.
tokenized_datasets = datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=["source", "target"]
)

# ── 4. MODEL ──────────────────────────────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# ── 5. METRICS (used only during final test evaluation) ───────────────────────
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_matches = sum(
        1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip()
    )
    em_score = exact_matches / max(len(decoded_preds), 1)

    try:
        chrf_score = sacrebleu.corpus_chrf(
            decoded_preds, [[l] for l in decoded_labels]
        ).score
    except Exception as e:
        print(f"chrF error: {e}")
        chrf_score = 0.0

    print(f"\n[TEST EVAL] EM: {em_score:.4f} | chrF: {chrf_score:.4f}", flush=True)
    return {"exact_match": em_score, "chrf": chrf_score}

# ── 5.5 CUSTOM CALLBACK FOR SAVING EVERY 5 EPOCHS ────────────────────────────
class SaveEveryFiveEpochsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Save at the end of every 5th epoch (e.g., 5, 10, 15...)
        if state.epoch > 0 and int(state.epoch) % 5 == 0:
            control.should_save = True
        return control

# ── 6. TRAINING ARGS — NO EVAL DURING TRAINING ───────────────────────────────
batch_size = 16

train_args = Seq2SeqTrainingArguments(
    output_dir="./byt5-gondi-translation",

    # ── Training ──────────────────────────────────────────
    num_train_epochs=50,
    per_device_train_batch_size=batch_size,
    learning_rate=1e-4,
    weight_decay=0.01,
    bf16=True,
    fp16=False,

    # ── Evaluation: DISABLED during training ──────────────
    eval_strategy="no",
    predict_with_generate=False,   # not needed during training

    # ── Logging ───────────────────────────────────────────
    logging_strategy="steps",
    logging_steps=100,            # More frequent logs
    logging_first_step=True,

    # ── Saving: Managed by our custom callback ───────────
    save_strategy="no",
    save_total_limit=5,           # Keep only last 2 checkpoints to save disk space

    # ── Windows deadlock fixes ────────────────────────────
    dataloader_num_workers=0,      # KEY: no DataLoader worker processes
    dataloader_pin_memory=False,   # no background pin-memory thread

    generation_max_length=64,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[SaveEveryFiveEpochsCallback()], # Register the 5-epoch save callback
)

# -- 7. MAIN -------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    print("\n--- GPU STATUS -------------------------------------------")
    is_cuda = torch.cuda.is_available()
    print(f"  CUDA Available : {is_cuda}")
    if is_cuda:
        print(f"  Device         : {torch.cuda.get_device_name(0)}")
        print(f"  Precision      : bfloat16")
    else:
        print("  WARNING: Running on CPU -- will be very slow!")
    print("----------------------------------------------------------\n")

    # -- Step 1: Train ---------------------------------------------------------
    print(f">>> STEP 1: Training started (Target: {train_args.num_train_epochs} epochs)...")
    
    # Check if we have a checkpoint to resume from
    last_checkpoint = None
    output_dir = train_args.output_dir
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Found checkpoint: {last_checkpoint}. Resuming training...")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    print(">>> STEP 1: Training complete!\n")

    # -- Step 2: Save model & tokenizer ---------------------------------------
    save_dir = "./byt5-gondi-final"
    print(f">>> STEP 2: Saving model and tokenizer to '{save_dir}'...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f">>> STEP 2: Model and tokenizer saved!\n")

    # -- Step 3: Test evaluation (generate + metrics) -------------------------
    print(">>> STEP 3: Running test evaluation (predict_with_generate=True)...")
    trainer.args.predict_with_generate = True
    trainer.compute_metrics = compute_metrics

    test_results = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        metric_key_prefix="test"
    )

    print("\n--- FINAL TEST RESULTS -----------------------------------")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("----------------------------------------------------------")
    print(">>> STEP 3: Done. Training pipeline complete!")
