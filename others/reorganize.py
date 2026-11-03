import os
import shutil

def main():
    base_dir = r"d:\New models Lokesh sir"
    
    # Create directories
    os.makedirs(os.path.join(base_dir, "Datasets"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "T5_Model"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "LSTM_Model"), exist_ok=True)
    
    # Define mappings
    files_to_move = {
        "Datasets": [
            "gondi_dataset_full.csv",
            "gondi_training_60k.csv",
            "gondi_training_dataset.csv",
            "extract_60k.py",
            "process_dataset.py"
        ],
        "T5_Model": [
            "t5_inference.py",
            "t5_train.py"
        ],
        "LSTM_Model": [
            "best_gondi_model.pt",
            "generate_cm.py",
            "gondi_bpe_model.json",
            "gondi_bpe_tokenizer.py",
            "gondi_seq2seq.py",
            "inference.py",
            "retrain_tokenizer.py",
            "train.py"
        ]
    }
    
    for folder, files in files_to_move.items():
        for f in files:
            src = os.path.join(base_dir, f)
            dst = os.path.join(base_dir, folder, f)
            if os.path.exists(src):
                print(f"Moving {f} to {folder}/")
                shutil.move(src, dst)
            
    # Update paths in the moved scripts
    
    # Update t5_train.py
    t5_train_path = os.path.join(base_dir, "T5_Model", "t5_train.py")
    if os.path.exists(t5_train_path):
        with open(t5_train_path, "r", encoding="utf-8") as file:
            content = file.read()
        content = content.replace('"gondi_training_60k.csv"', '"../Datasets/gondi_training_60k.csv"')
        with open(t5_train_path, "w", encoding="utf-8") as file:
            file.write(content)
            
    # Update LSTM train.py
    lstm_train_path = os.path.join(base_dir, "LSTM_Model", "train.py")
    if os.path.exists(lstm_train_path):
        with open(lstm_train_path, "r", encoding="utf-8") as file:
            content = file.read()
        content = content.replace('"gondi_training_60k.csv"', '"../Datasets/gondi_training_60k.csv"')
        with open(lstm_train_path, "w", encoding="utf-8") as file:
            file.write(content)

    # Update retrain_tokenizer.py
    retrain_path = os.path.join(base_dir, "LSTM_Model", "retrain_tokenizer.py")
    if os.path.exists(retrain_path):
        with open(retrain_path, "r", encoding="utf-8") as file:
            content = file.read()
        content = content.replace('"gondi_training_60k.csv"', '"../Datasets/gondi_training_60k.csv"')
        with open(retrain_path, "w", encoding="utf-8") as file:
            file.write(content)

    print("Reorganization complete! You can delete this reorganize.py script.")

if __name__ == "__main__":
    main()
