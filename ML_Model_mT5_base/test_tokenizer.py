import os
import sys

# Change directory to the model directory
os.chdir(r"d:\ML MODEL mT5 base")

from transformers import AutoTokenizer

model_dir = "./byt5-gondi-final"
if not os.path.isdir(model_dir):
    model_dir = "google/byt5-small"

print(f"Loading tokenizer from: {model_dir}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

sample_text = "Translate English to Gondi: Hello"
print(f"Original Text: '{sample_text}'\n")

tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)

print("Tokens (Bytes):")
print(tokens)
print("\nToken IDs:")
print(token_ids)

print("\nDecoded back to text:")
print(tokenizer.decode(token_ids, skip_special_tokens=True))
