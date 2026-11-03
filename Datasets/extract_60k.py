import pandas as pd
import random

def extract_best_60k():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, 'gondi_training_dataset.csv')
    output_file = os.path.join(BASE_DIR, 'gondi_training_60k.csv')
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    initial_count = len(df)
    print(f"Total rows: {initial_count}")
    
    # 1. Drop completely empty rows or rows with missing values
    df = df.dropna(subset=['Gondi', 'Hindi'])
    
    # 2. Drop exact duplicates (keeps only unique sentences)
    df = df.drop_duplicates(subset=['Gondi'])
    df = df.drop_duplicates(subset=['Hindi'])
    print(f"Rows after removing duplicates & NaNs: {len(df)}")
    
    # Ensure columns are strings
    df['Gondi'] = df['Gondi'].astype(str).str.strip()
    df['Hindi'] = df['Hindi'].astype(str).str.strip()
    
    # Calculate word counts
    df['gondi_len'] = df['Gondi'].apply(lambda x: len(x.split()))
    df['hindi_len'] = df['Hindi'].apply(lambda x: len(x.split()))
    
    # 3. Filter by length: Minimum 3 words, Maximum 25 words (best for seq2seq training)
    # 1-2 word sentences are often just vocabulary mappings, >25 words are too complex for initial learning
    good_df = df[
        (df['gondi_len'] >= 3) & (df['gondi_len'] <= 25) &
        (df['hindi_len'] >= 3) & (df['hindi_len'] <= 25)
    ].copy()
    
    # 4. Filter by Length Ratio: 
    # A sentence in Gondi shouldn't be 5 times longer than Hindi. Keep ratio under 2.0.
    good_df['len_ratio'] = good_df.apply(
        lambda row: max(row['gondi_len'], row['hindi_len']) / max(min(row['gondi_len'], row['hindi_len']), 1), 
        axis=1
    )
    good_df = good_df[good_df['len_ratio'] <= 2.0]
    
    print(f"High Quality rows remaining after strict length & ratio filtering: {len(good_df)}")
    
    # 5. Extract exactly 60,000 (if we have more than 60k, we take a random sample)
    if len(good_df) > 60000:
        print("Sampling 60,000 from the high quality pool...")
        final_df = good_df.sample(n=60000, random_state=42)
    else:
        print(f"We only have {len(good_df)} high quality rows. Taking all of them.")
        final_df = good_df
        
    # Drop the temporary calculation columns
    final_df = final_df.drop(columns=['gondi_len', 'hindi_len', 'len_ratio'])
    
    # Save the new 60k dataset
    print(f"Saving final dataset to {output_file}...")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✅ Best quality 60K dataset extracted successfully!")

if __name__ == "__main__":
    extract_best_60k()
