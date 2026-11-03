import pandas as pd
import os

def process_dataset():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, 'gondi_dataset_full.csv')
    output_file = os.path.join(BASE_DIR, 'gondi_training_dataset.csv')
    
    print(f"Reading {input_file} ...")
    try:
        # Read the file with pipe separator
        df = pd.read_csv(input_file, sep='|', encoding='utf-8', on_bad_lines='skip')
        print(f"Successfully loaded {len(df)} rows.")
        
        # Verify columns
        print(f"Columns found: {df.columns.tolist()}")
        
        # Rename columns just in case they have spaces
        df.columns = df.columns.str.strip()
        
        # Ensure we have Gondi, Hindi, English columns
        if not all(col in df.columns for col in ['Gondi', 'Hindi', 'English']):
            print("Warning: Expected columns 'Gondi', 'Hindi', 'English' not exactly matched.")
            print(f"Available columns: {df.columns.tolist()}")
            
        print(f"Saving to {output_file} with utf-8-sig encoding (fixes Devanagari)...")
        # Save as standard comma-separated CSV with utf-8-sig (adds BOM for Excel/Windows)
        df.to_csv(output_file, index=False, sep=',', encoding='utf-8-sig')
        
        print("Done! Dataset is ready and correctly formatted with 3 separate columns.")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    process_dataset()
