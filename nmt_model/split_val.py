import os
import pandas as pd

def main():
    base_dir = "d:/New models Lokesh sir/nmt_model/Data set"
    val_path = os.path.join(base_dir, "val.csv")
    
    print(f"Loading {val_path}...")
    df = pd.read_csv(val_path)
    
    total_len = len(df)
    part_size = total_len // 10
    print(f"Total validation samples: {total_len}. Size per fold: {part_size}")
    
    for i in range(10):
        start_idx = i * part_size
        # For the last fold, include the remainder
        end_idx = (i + 1) * part_size if i < 9 else total_len
        
        fold_df = df.iloc[start_idx:end_idx]
        fold_path = os.path.join(base_dir, f"val{i+1}.csv")
        fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
        print(f"Saved fold {i+1} ({len(fold_df)} samples) -> {fold_path}")

    # Leak check
    print("\nVerifying no overlap (zero leaks) between validation folds...")
    folds_data = []
    for i in range(10):
        df_fold = pd.read_csv(os.path.join(base_dir, f"val{i+1}.csv"))
        folds_data.append(set(df_fold["Gondi"].astype(str).tolist()))
        
    overlap_found = False
    for i in range(10):
        for j in range(i + 1, 10):
            intersection = folds_data[i].intersection(folds_data[j])
            if intersection:
                print(f"WARNING: Overlap of {len(intersection)} samples between val{i+1} and val{j+1}!")
                overlap_found = True
                
    if not overlap_found:
        print("[SUCCESS] Zero leakage verified between all 10 folds.")

if __name__ == "__main__":
    main()
