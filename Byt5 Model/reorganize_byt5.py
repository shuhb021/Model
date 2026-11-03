import os
import shutil

base_dir = r"d:\New models Lokesh sir\Byt5 Model"

# 1. Define new folder paths
model_dest = os.path.join(base_dir, "Model_Weights")
tokenizer_dest = os.path.join(base_dir, "Tokenizer_Files")
results_dest = os.path.join(base_dir, "Evaluation_Results")
scripts_dest = os.path.join(base_dir, "Scripts")

for folder in [model_dest, tokenizer_dest, results_dest, scripts_dest]:
    os.makedirs(folder, exist_ok=True)

# 2. Move Model Files
# If byt5_gondi_model exists, move its contents to Model_Weights
model_src = os.path.join(base_dir, "byt5_gondi_model")
if os.path.exists(model_src):
    for item in os.listdir(model_src):
        shutil.move(os.path.join(model_src, item), os.path.join(model_dest, item))
    os.rmdir(model_src)

# 3. Move Tokenizer Files
tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "gondi_bpe_tokenizer.py"]
for f in tokenizer_files:
    src = os.path.join(base_dir, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(tokenizer_dest, f))

# 4. Move Result Files
result_extensions = [".png", ".json", ".csv"]
for f in os.listdir(base_dir):
    if any(f.endswith(ext) for ext in result_extensions) and "byt5" not in f:
        # Avoid moving scripts that might have .json (none here)
        shutil.move(os.path.join(base_dir, f), os.path.join(results_dest, f))
# Specifically move predictions_500.csv and evaluation_results.json
for f in ["predictions_500.csv", "evaluation_results.json"]:
    src = os.path.join(base_dir, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(results_dest, f))

# 5. Move scripts
scripts = ["byt5_train.py", "byt5_evaluate.py", "byt5_inference.py"]
for f in scripts:
    src = os.path.join(base_dir, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(scripts_dest, f))

# 6. Cleanup unnecessary files
unnecessary = ["debug_tokenizer.py", "fix_merges.py"]
for f in unnecessary:
    src = os.path.join(base_dir, f)
    if os.path.exists(src):
        os.remove(src)

# Remove training artifacts folder if it exists
train_results = os.path.join(base_dir, "byt5_gondi_results")
if os.path.exists(train_results):
    shutil.rmtree(train_results)

print("Reorganization complete!")
