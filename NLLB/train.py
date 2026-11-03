import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from model import CustomNllbModel, ModelConfig
from inference import calculate_chrf
from gondi_bpe_tokenizer import GondiTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSLATION DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ParallelTranslationDataset(Dataset):
    """
    Loads parallel sentences from CSV and tokenises them.
    Prepends language tags: source text and target text.
    """
    def __init__(self, csv_path: str, tokenizer: GondiTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data: List[Tuple[str, str]] = []
        
        # Load dataset
        if os.path.exists(csv_path):
            print(f"Loading dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            # Try to find Gondi and Hindi/English columns
            gondi_cols = [c for c in df.columns if "gondi" in c.lower()]
            hindi_cols = [c for c in df.columns if "hindi" in c.lower() or "hin" in c.lower()]
            english_cols = [c for c in df.columns if "english" in c.lower() or "eng" in c.lower()]
            
            src_col = gondi_cols[0] if gondi_cols else df.columns[0]
            tgt_col = hindi_cols[0] if hindi_cols else (english_cols[0] if english_cols else df.columns[1])
            
            for _, row in df.iterrows():
                src_val = str(row[src_col]).strip()
                tgt_val = str(row[tgt_col]).strip()
                if src_val and tgt_val:
                    self.data.append((src_val, tgt_val))
        else:
            print(f"Dataset path {csv_path} not found. Generating synthetic parallel lines.")
            # Fallback to dummy data
            dummy_gondi = [
                "aggator vat itteke nahan",
                "an akkal adenlasi pator",
                "gondi pargoti adenla manval",
                "vatusval akkurpok"
            ] * 25
            dummy_hindi = [
                "वह आदमी वहां गया था",
                "उसने बुद्धि से काम किया",
                "गोंडी भाषा बहुत पुरानी है",
                "धूप में सूखा हुआ फल"
            ] * 25
            for s, t in zip(dummy_gondi, dummy_hindi):
                self.data.append((s, t))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.data[idx]
        
        # Tokenize source & target
        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)
        
        # Pad or truncate
        # Add BOS ([START]) and EOS ([END])
        start_id = self.tokenizer.token_to_id.get("[START]", 2)
        end_id = self.tokenizer.token_to_id.get("[END]", 3)
        pad_id = self.tokenizer.token_to_id.get("[PAD]", 0)

        src_ids = [start_id] + src_ids[:self.max_len - 2] + [end_id]
        tgt_ids = [start_id] + tgt_ids[:self.max_len - 2] + [end_id]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLLATOR WITH PADDING
# ─────────────────────────────────────────────────────────────────────────────

class PaddingCollator:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_tensors, tgt_tensors = zip(*batch)
        
        # Pad source sequences
        src_padded = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=self.pad_id)
        # Pad target sequences
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=self.pad_id)
        
        # Attention masks
        # 1 for valid tokens, 0 for pad tokens
        # shape: (Batch, 1, 1, SeqLen_src)
        src_mask = (src_padded != self.pad_id).unsqueeze(1).unsqueeze(2).to(torch.float32)
        
        # Target masks: causal + padding combined
        # shape: (Batch, 1, SeqLen_tgt, SeqLen_tgt)
        batch_size, tgt_len = tgt_padded.shape
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len))).view(1, 1, tgt_len, tgt_len)
        pad_mask = (tgt_padded != self.pad_id).unsqueeze(1).unsqueeze(2).to(torch.float32)
        tgt_mask = causal_mask * pad_mask
        
        return src_padded, tgt_padded, src_mask, tgt_mask

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: CustomNllbModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accumulation_steps: int = 4
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (src_ids, tgt_ids, src_mask, tgt_mask) in enumerate(dataloader):
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        # Slice targets for teacher forcing:
        # Inputs to decoder are tokens from start to N-1
        # Outputs to predict are tokens from 1 to N (shifted left)
        dec_inputs = tgt_ids[:, :-1]
        dec_labels = tgt_ids[:, 1:]
        
        # Slice target mask accordingly
        dec_mask = tgt_mask[:, :, :-1, :-1]

        # Forward pass with AMP mixed-precision
        with torch.cuda.amp.autocast():
            logits = model(src_ids, dec_inputs, src_mask=src_mask, tgt_mask=dec_mask)
            
            # Cross entropy loss ignoring padding tokens
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), dec_labels.reshape(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        if (step + 1) % grad_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accumulation_steps

    return total_loss / len(dataloader)

# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Avoid print encoding crashes on Windows PowerShell
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = GondiTokenizer.load(tokenizer_path)
    else:
        # Fallback to local tokenizer training or load from project
        print("Tokenizer JSON not found in NLLB folder, building a mock BPE tokenizer config...")
        tokenizer = GondiTokenizer()
        tokenizer.train(["gondi text sample", "hindi text translation"], vocab_size=300, verbose=False)

    # 2. Config & Model
    # Budgeted parameter count ~137M parameters (vocab=32000, layers=8/4, heads=8, hidden=512)
    config = ModelConfig(
        vocab_size=max(32000, tokenizer.vocab_size()),
        d_model=512,
        d_ff=2048,
        num_encoder_layers=8,
        num_decoder_layers=4,
        num_heads=8,
        dropout=0.15,
        use_lora=True
    )
    
    model = CustomNllbModel(config).to(device)
    print(f"Model initialized with vocabulary size: {config.vocab_size}")

    # 3. Loader & Collator
    csv_path = "../Datasets/gondi_training_60k.csv"
    dataset = ParallelTranslationDataset(csv_path, tokenizer, max_len=128)
    collator = PaddingCollator(pad_id=0)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Small batch size for demonstration/RTX budget
        shuffle=True,
        collate_fn=collator
    )

    # 4. Optimizer & Scaler
    # AdamW with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler()

    # 5. Small train verification loop (2 epochs)
    print("\nStarting small train verification...")
    for epoch in range(1, 3):
        loss = train_epoch(model, dataloader, optimizer, scaler, device)
        print(f"Epoch {epoch}/2 | Training Loss: {loss:.4f}")

    # 6. Save checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "nllb_asym_checkpoint.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, checkpoint_path)
    print(f"\nCheckpoint successfully saved → {checkpoint_path}")
    print("Verification loop completed successfully!")

if __name__ == "__main__":
    main()
