import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "t5_gondi_model")
    print("=" * 60)
    print(f" Loading T5 model and BPE tokenizer from {model_path}...")
    print("=" * 60)
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have run 'python t5_train.py' successfully and the model is saved.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    src_col = "Gondi"
    tgt_col = "English" # Make sure this matches what was used during training
    prefix = f"translate {src_col} to {tgt_col}: "
    
    print("\n--- Gondi to English Translator (T5 + Fast BPE Tokenizer) ---")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            gondi_text = input("Enter Gondi sentence: ").strip()
        except EOFError:
            break
            
        if gondi_text.lower() in ['quit', 'exit']:
            break
        if not gondi_text:
            continue
            
        # Add the task prefix to the input
        input_text = prefix + gondi_text
        
        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Generate the translation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30,  # Prevent generating infinitely long text
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5, # Heavily penalize repeating the same words
                length_penalty=1.0
            )
            
        # Decode the generated token IDs back to a string
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Translation: {translation}\n")

if __name__ == "__main__":
    main()
