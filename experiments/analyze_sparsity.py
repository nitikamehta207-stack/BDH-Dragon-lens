"""
BDH Microscope - Sparsity Analysis
=====================================
Ye script BDHInstrumented model use karke:
1. Random input dekar model chalata hai (bina training ke)
2. Har layer/level ka sparsity nikalta hai
3. Results ko CSV mein save karta hai (dashboard ke liye)
"""

import torch
import pandas as pd
import os

from bdh_instrumented import BDHInstrumented, BDHConfig


def calculate_sparsity(tensor):
    """Kitne % values zero hain (ya bahut chhote, near-zero)"""
    total = tensor.numel()
    zeros = (tensor == 0).sum().item()
    return round((zeros / total) * 100, 2)


def main():
    print("=" * 60)
    print("BDH MICROSCOPE - Sparsity Analysis")
    print("=" * 60)

    # STEP 1: Config aur model banao (abhi RANDOM weights, training nahi ki)
    config = BDHConfig()  # default: 6 layers, 256 embd, 4 heads
    model = BDHInstrumented(config)
    model.eval()  # inference mode

    print(f"\nModel config: {config}")

    # STEP 2: Sample input banao
    # BDH byte-level hai (vocab_size=256), to koi bhi text bytes se convert karke
    # ya seedha random integers 0-255 se test kar sakte ho
    sample_text = "BDH is a post-transformer architecture"
    sample_bytes = list(sample_text.encode('utf-8'))  # text ko bytes mein convert
    idx = torch.tensor([sample_bytes])  # shape: (1, T)

    print(f"\nInput text: '{sample_text}'")
    print(f"Input shape: {idx.shape} (bytes: {sample_bytes[:10]}...)")

    # STEP 3: Forward pass chalao (activations automatically capture ho jayenge)
    with torch.no_grad():
        logits, loss = model(idx)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Levels captured: {len(model.captured_activations)}")

    # STEP 4: Har level ki sparsity nikalo
    results = []
    for layer_data in model.captured_activations:
        level = layer_data['level']

        x_sparsity = calculate_sparsity(layer_data['x_sparse'])
        y_sparsity = calculate_sparsity(layer_data['y_sparse'])
        xy_sparsity = calculate_sparsity(layer_data['xy_sparse'])

        results.append({
            'level': level,
            'x_sparse_pct': x_sparsity,
            'y_sparse_pct': y_sparsity,
            'xy_sparse_pct (gated)': xy_sparsity,
            'x_sparse_shape': str(list(layer_data['x_sparse'].shape)),
        })

        print(f"\nLevel {level}:")
        print(f"  x_sparse (input->encoder->relu):  {x_sparsity}% sparse")
        print(f"  y_sparse (attn->encoder_v->relu):  {y_sparsity}% sparse")
        print(f"  xy_sparse (gated, x*y):             {xy_sparsity}% sparse")

    # STEP 5: Results save karo
    df = pd.DataFrame(results)
    os.makedirs('../results', exist_ok=True)
    df.to_csv('../results/sparsity_results.csv', index=False)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved to results/sparsity_results.csv")

    print("\nNOTE (Research Contract ke liye):")
    print("  Ye ek UNTRAINED (random weights) model pe MEASURED hai.")
    print("  Paper ka ESTABLISHED claim: trained models mein sparse,")
    print("  predictability-linked activity dikhti hai.")
    print("  Trained model pe zyada meaningful hoga - agar time bache")
    print("  to train.py chalake ye script dobara chalao.")

    return df


if __name__ == "__main__":
    main()
