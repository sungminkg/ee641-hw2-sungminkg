"""
Analysis and evaluation experiments for trained GAN models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from metrics import *
from visualize import *

def interpolation_experiment(generator, device):
    """
    Interpolate between latent codes to generate smooth transitions.
    
    TODO:
    1. Find latent codes for specific letters (via optimization)
    2. Interpolate between them
    3. Visualize the path from A to Z
    """
    generator.eval()
    z_dim = generator.z_dim

    z_start = torch.randn(1, z_dim, device=device)
    z_end = torch.randn(1, z_dim, device=device)

    # linear interpolation
    alphas = torch.linspace(0, 1, steps=10).to(device)
    zs = [(1 - a) * z_start + a * z_end for a in alphas]

    # generate and visualize
    fig, axes = plt.subplots(1, len(zs), figsize=(15, 3))
    with torch.no_grad():
        for i, z in enumerate(zs):
            img = generator(z).squeeze().cpu()
            img = (img + 1) / 2
            axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i].axis("off")
    plt.suptitle("Latent Interpolation", fontsize=14)
    return fig

def style_consistency_experiment(conditional_generator, device):
    """
    Test if conditional GAN maintains style across letters.
    
    TODO:
    1. Fix a latent code z
    2. Generate all 26 letters with same z
    3. Measure style consistency
    """
    conditional_generator.eval()
    z_dim = conditional_generator.z_dim
    z = torch.randn(1, z_dim, device=device)

    generated = {}
    with torch.no_grad():
        for i in range(26):
            label = torch.zeros(1, 26, device=device)
            label[0, i] = 1
            img = conditional_generator(z, label)
            generated.setdefault(i, []).append(img)
    score = font_consistency_score(generated)

    fig = plot_alphabet_grid(conditional_generator, device=device, z_dim=z_dim, seed=42)
    plt.suptitle(f"Style Consistency (score={score:.2f})", fontsize=16)
    return fig, score

def mode_recovery_experiment(generator_checkpoints):
    """
    Analyze how mode collapse progresses and potentially recovers.
    
    TODO:
    1. Load checkpoints from different epochs
    2. Measure mode coverage at each checkpoint
    3. Identify when specific letters disappear/reappear
    """
    coverages = {}
    for epoch, gen in generator_checkpoints.items():
        gen.eval()
        z_dim = gen.z_dim
        with torch.no_grad():
            z = torch.randn(500, z_dim, device=next(gen.parameters()).device)
            fake_imgs = gen(z)
        result = mode_coverage_score(fake_imgs)
        coverages[epoch] = result

    # Plot coverage score over epochs
    fig = plt.figure(figsize=(8, 5))
    epochs = sorted(coverages.keys())
    scores = [coverages[e]["coverage_score"] for e in epochs]
    plt.plot(epochs, scores, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mode Coverage")
    plt.title("Mode Recovery Over Epochs")
    plt.ylim([0, 1.1])
    plt.grid(True)
    return fig, coverages


if __name__ == "__main__":
    import os
    from pathlib import Path
    from models import Generator

    # Load checkpoint and config
    exp_tag = "fixed_feature_matching"  # "vanilla or "fixed_feature_matching"
    ckpt_path = f"results/best_generator_{exp_tag}.pth"
    out_dir = f"results/visualizations"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    device = config["device"]

    generator = Generator(z_dim=config["z_dim"]).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()


    # 1. Interpolation
    fig = interpolation_experiment(generator, device=device)
    fig.savefig(os.path.join(out_dir, f"interpolation_{exp_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Style Consistency (if conditional GAN)
    if hasattr(generator, "conditional") and generator.conditional:
        fig, score = style_consistency_experiment(generator, device=device)
        fig.savefig(os.path.join(out_dir, f"style_consistency_{exp_tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        with open(os.path.join(out_dir, f"style_score_{exp_tag}.txt"), "w") as f:
            f.write(f"Style consistency score: {score:.4f}\n")

    # 3. Mode Coverage Histogram 
    with torch.no_grad():
        z = torch.randn(500, config["z_dim"], device=device)
        fake_imgs = generator(z)

    result = mode_coverage_score(fake_imgs)
    coverage_score = result["coverage_score"]

    letter_counts = result["letter_counts"]
    per_class = [letter_counts.get(i, 0) for i in range(26)]  # frequency for each letter A–Z

    # Plot histogram of which letters survived
    fig = plt.figure(figsize=(10, 5))
    letters = [chr(65 + i) for i in range(26)]  # A–Z
    plt.bar(letters, per_class, color="skyblue")
    plt.title(f"Mode Coverage Histogram ({exp_tag}), Score={coverage_score:.2f}")
    plt.ylabel("Frequency")
    plt.xlabel("Letters (A–Z)")
    plt.ylim([0, max(per_class) + 1])
    plt.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"mode_coverage_hist_{exp_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Evaluation complete. Results saved to {out_dir}/")
