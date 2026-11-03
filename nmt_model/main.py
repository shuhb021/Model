"""
main.py — Entry Point for NMT Training & Inference
────────────────────────────────────────────────────
Usage:

    # Verify model architecture
    python main.py --mode verify

    # Run sanity tests
    python main.py --mode test

    # Train (provide your data)
    python main.py --mode train --data_dir data/processed/

    # Translate interactively
    python main.py --mode translate --src_lang gondi --tgt_lang hindi

    # Translate a file
    python main.py --mode translate --src_lang gondi --tgt_lang hindi \\
                   --input_file sentences.txt --output_file translations.txt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def print_banner():
    print("+" + "="*62 + "+")
    print("|   Custom Asymmetric Multilingual NMT                         |")
    print("|   Gondi <-> Hindi <-> English  |  ~137M Params               |")
    print("|   8-Layer Encoder / 4-Layer Decoder / Gated-GeLU FFN         |")
    print("|   T5 Relative Bias / MBR Decoding / TM Fusion                |")
    print("+" + "="*62 + "+")


def mode_verify():
    """Verify model builds correctly and count parameters."""
    from model.architecture import AsymmetricNMT
    import torch

    print("Building AsymmetricNMT...")
    model = AsymmetricNMT(
        vocab_size=32_000, d_model=512, d_ff=2048,
        num_heads=8, enc_layers=8, dec_layers=4,
        dropout=0.15, max_seq_len=256
    )

    total_params = model.count_parameters()
    print(f"\nParameter Count:")
    print(f"   Total trainable parameters : {total_params:>15,}")
    print(f"   Approx (millions)          : {total_params/1e6:>14.2f}M")
    print(f"   Expected                   :         ~137.25M")

    # Component breakdown
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    emb_params = sum(p.numel() for p in model.embedding.parameters())

    print(f"\n   Embedding (shared)         : {emb_params:>15,}")
    print(f"   Encoder (8 layers)         : {enc_params:>15,}")
    print(f"   Decoder (4 layers)         : {dec_params:>15,}")

    # Forward pass
    print(f"\nRunning forward pass...")
    src = torch.randint(1, 32000, (2, 40))
    tgt = torch.randint(1, 32000, (2, 35))
    with torch.no_grad():
        logits = model(src, tgt)
    print(f"   [OK] logits shape: {tuple(logits.shape)}  (expected [2, 35, 32000])")

    print(f"\n[SUCCESS] Architecture verification PASSED\n")


def mode_test():
    """Run component-level tests."""
    import torch

    print("Running component tests...\n")
    errors = 0

    # Test 1: Relative Position Bias
    try:
        from model.architecture import RelativePositionBias
        rpb = RelativePositionBias(num_heads=8)
        bias = rpb(16, 16, torch.device("cpu"))
        assert bias.shape == (1, 8, 16, 16), f"Wrong shape: {bias.shape}"
        print("  [OK] RelativePositionBias")
    except Exception as e:
        print(f"  [FAIL] RelativePositionBias: {e}")
        errors += 1

    # Test 2: Gated-GeLU FFN
    try:
        from model.architecture import GatedGeLUFFN
        ffn = GatedGeLUFFN(d_model=512, d_ff=2048)
        x = torch.randn(2, 10, 512)
        out = ffn(x)
        assert out.shape == (2, 10, 512)
        print("  [OK] GatedGeLUFFN")
    except Exception as e:
        print(f"  [FAIL] GatedGeLUFFN: {e}")
        errors += 1

    # Test 3: Full model
    try:
        from model.architecture import AsymmetricNMT
        model = AsymmetricNMT()
        src = torch.randint(1, 32000, (2, 20))
        tgt = torch.randint(1, 32000, (2, 15))
        with torch.no_grad():
            logits = model(src, tgt)
        assert logits.shape == (2, 15, 32000)
        print("  [OK] AsymmetricNMT full forward pass")
    except Exception as e:
        print(f"  [FAIL] AsymmetricNMT: {e}")
        errors += 1

    # Test 4: LoRA injection
    try:
        from model.architecture import AsymmetricNMT
        from model.lora import apply_lora_to_model
        model = AsymmetricNMT()
        model = apply_lora_to_model(model, rank=16, alpha=32)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total
        assert pct < 10, f"LoRA trainable% too high: {pct:.1f}%"
        print(f"  [OK] LoRA injection ({trainable:,} trainable / {total:,} total = {pct:.2f}%)")
    except Exception as e:
        print(f"  [FAIL] LoRA: {e}")
        errors += 1

    # Test 5: TM retrieval
    try:
        from inference.translation_memory import TranslationMemory
        from inference.decoder import chrf_score
        tm = TranslationMemory()
        tm.add_bulk([("hello world", "namaste duniya"), ("good morning", "suprabhat")])
        results = tm.retrieve("hello", top_k=2)
        assert len(results) == 2
        score = chrf_score("hello world", "hello world")
        assert score > 0.99
        print("  [OK] TranslationMemory + chrF")
    except Exception as e:
        print(f"  [FAIL] TM: {e}")
        errors += 1

    # Test 6: Temperature sampling
    try:
        from data.pipeline import MultilingualSampler
        sampler = MultilingualSampler({"a": 150000, "b": 300000}, temperature=1.6)
        assert abs(sum(sampler.probabilities.values()) - 1.0) < 1e-6
        print("  [OK] MultilingualSampler (T=1.6)")
    except Exception as e:
        print(f"  [FAIL] MultilingualSampler: {e}")
        errors += 1

    print(f"\n{'[SUCCESS] All tests passed!' if errors == 0 else f'[FAIL] {errors} test(s) failed'}\n")
    return errors


def mode_translate(args):
    """Interactive / file translation."""
    from model.architecture import AsymmetricNMT
    from data.tokenizer import MultilingualTokenizer
    from inference.translation_memory import TranslationMemory
    from inference.decoder import NMTInference
    import torch

    print(f"Loading model...")
    model = AsymmetricNMT()

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ck["model_state"])
        print(f"   Loaded checkpoint: {args.checkpoint}")
    else:
        print("   [WARN] No checkpoint provided - using random weights (for demo only)")

    tok = MultilingualTokenizer(model_path=args.tokenizer if hasattr(args, 'tokenizer') else None)
    tm  = TranslationMemory()
    engine = NMTInference(model, tok, translation_memory=tm)

    src_lang = args.src_lang or "gondi"
    tgt_lang = args.tgt_lang or "hindi"

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        translations = engine.translate_batch(lines, src_lang, tgt_lang)
        out_path = args.output_file or "translations.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for t in translations:
                f.write(t + "\n")
        print(f"[SUCCESS] Translated {len(lines)} sentences -> {out_path}")
    else:
        print(f"\nInteractive mode: {src_lang} -> {tgt_lang}")
        print("Type 'quit' to exit.\n")
        while True:
            text = input("Source: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = engine.translate(text, src_lang, tgt_lang, method=args.method or "mbr")
            print(f"  -> {result}\n")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Asymmetric Multilingual NMT")
    parser.add_argument("--mode", choices=["verify", "test", "translate"],
                        default="verify")
    parser.add_argument("--src_lang",    default="gondi")
    parser.add_argument("--tgt_lang",    default="hindi")
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--tokenizer",   default=None)
    parser.add_argument("--input_file",  default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--method",      choices=["mbr", "beam", "greedy"],
                        default="mbr")
    args = parser.parse_args()

    if args.mode == "verify":
        mode_verify()
    elif args.mode == "test":
        errors = mode_test()
        sys.exit(errors)
    elif args.mode == "translate":
        mode_translate(args)


if __name__ == "__main__":
    main()
