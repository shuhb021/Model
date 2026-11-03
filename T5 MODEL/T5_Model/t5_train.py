import pandas as pd
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

def train_bpe_tokenizer(df, vocab_size=5000):
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer using the Hugging Face `tokenizers` library.
    This creates a fast BPE tokenizer compatible with the T5 model.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Define special tokens expected by T5
    special_tokens = ["[UNK]", "[PAD]", "</s>", "[START]"]
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    
    def get_training_corpus():
        # Iterate over both Gondi and English/Hindi texts for a combined vocabulary
        for i in range(0, len(df), 1000):
            yield df["Gondi"].dropna()[i : i + 1000].astype(str).tolist() + \
                  df["English"].dropna()[i : i + 1000].astype(str).tolist()
            
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Wrap into a Fast Tokenizer for transformers
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="</s>",
        bos_token="[START]"
    )
    return fast_tokenizer

def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Since t5_train.py is in 'T5 MODEL/T5_Model', Datasets is two levels up
    csv_path = os.path.join(base_dir, "..", "..", "Datasets", "gondi_training_60k.csv")
    src_col = "Gondi"
    tgt_col = "English" # Change to "Hindi" if you want to translate to Hindi
    
    print("=" * 60)
    print(f" Loading data from {csv_path}...")
    print("=" * 60)
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(subset=[src_col, tgt_col]).reset_index(drop=True)
    
    # Convert all content to string
    df[src_col] = df[src_col].astype(str)
    df[tgt_col] = df[tgt_col].astype(str)
    
    print(f"Total sentences for training: {len(df)}")
    
    print("\nTraining BPE Tokenizer...")
    tokenizer = train_bpe_tokenizer(df, vocab_size=5000)
    tokenizer.save_pretrained(os.path.join(base_dir, "..", "t5_bpe_tokenizer"))
    print("✅ Tokenizer saved to t5_bpe_tokenizer")

    # 1. Prepare Dataset
    dataset = Dataset.from_pandas(df[[src_col, tgt_col]])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    prefix = f"translate {src_col} to {tgt_col}: "
    max_length = 64
    
    def preprocess_function(examples):
        # We MUST append the EOS token so the model learns when to stop generating
        inputs = [prefix + str(ex) + " </s>" for ex in examples[src_col]]
        targets = [str(ex) + " </s>" for ex in examples[tgt_col]]
        
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        
        # Replace pad_token_id with -100 so it's ignored in loss computation
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing the dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=[src_col, tgt_col])
    
    # 2. Initialize T5 Model
    print("\nInitializing a T5 model from scratch...")
    # T5 models expect small vocabularies and standard dimensionalities. 
    # For a 60k dataset, a small T5 configuration will train quickly and effectively.
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=256,
        d_ff=1024,
        d_kv=64,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    model = T5ForConditionalGeneration(config)
    
    # 3. Training Arguments & Trainer Setup
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(base_dir, "t5_gondi_results"),
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Automatically use mixed precision if GPU is available
        logging_steps=100,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("\n" + "=" * 60)
    print(" Starting T5 training loop...")
    print("=" * 60)
    trainer.train()
    
    # 4. Save Final Model
    print("\nSaving final model...")
    model_save_path = os.path.join(base_dir, "..", "t5_gondi_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("✅ Done! T5 model and BPE tokenizer saved to t5_gondi_model")

if __name__ == "__main__":
    main()
