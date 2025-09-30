"""
Latent space analysis tools for hierarchical VAE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE
from pathlib import Path
from visualize import *

def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """
    model.eval()
    all_z_high, all_z_low, all_labels = [], [], []

    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            out = model(patterns)
            mu_low, mu_high = out['mu_low'], out['mu_high']
            all_z_low.append(mu_low.cpu().numpy())
            all_z_high.append(mu_high.cpu().numpy())
            all_labels.append(styles.numpy())

    z_low = np.concatenate(all_z_low, axis=0)
    z_high = np.concatenate(all_z_high, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # t-SNE for high-level latent
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_high_2d = tsne.fit_transform(z_high)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_high_2d[:, 0], z_high_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Style label")
    plt.title("t-SNE of High-Level Latent (z_high)")
    plt.show()

    return z_high, z_low, labels

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    model.eval()
    with torch.no_grad():
        p1, p2 = pattern1.to(device).unsqueeze(0), pattern2.to(device).unsqueeze(0)
        out1, out2 = model(p1), model(p2)
        z_high1, z_low1 = out1['mu_high'], out1['mu_low']
        z_high2, z_low2 = out2['mu_high'], out2['mu_low']

        high_interps, low_interps = [], []

        for alpha in np.linspace(0, 1, n_steps):
            z_high = (1 - alpha) * z_high1 + alpha * z_high2
            z_low = (1 - alpha) * z_low1 + alpha * z_low2
            logits = model.decode_hierarchy(z_high, z_low)
            pattern = torch.sigmoid(logits).cpu().squeeze(0).numpy()
            high_interps.append(pattern)

        return high_interps

def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    model.eval()
    z_high_by_genre, z_low_by_genre = {}, {}

    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            out = model(patterns)
            mu_high, mu_low = out['mu_high'].cpu().numpy(), out['mu_low'].cpu().numpy()
            for s, zh, zl in zip(styles.numpy(), mu_high, mu_low):
                z_high_by_genre.setdefault(s, []).append(zh)
                z_low_by_genre.setdefault(s, []).append(zl)

    # Compute variances
    genre_means = [np.mean(z_high_by_genre[g], axis=0) for g in z_high_by_genre]
    overall_high_var = np.var(np.vstack(genre_means), axis=0).mean()

    within_high_var = np.mean([np.var(z_high_by_genre[g], axis=0).mean() for g in z_high_by_genre])
    within_low_var = np.mean([np.var(z_low_by_genre[g], axis=0).mean() for g in z_low_by_genre])

    return {
        "z_high_within_var": float(within_high_var),
        "z_high_between_var": float(overall_high_var),
        "z_low_within_var": float(within_low_var),
        "disentanglement_score": float(overall_high_var / (within_high_var + 1e-8))
    }

def controllable_generation(model, genre_labels, device='cuda'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    model.eval()
    patterns_by_genre = {}

    with torch.no_grad():
        for genre in genre_labels:
            # Assume each genre maps to a point in z_high space
            z_high = torch.randn(1, model.z_high_dim).to(device)
            genre_patterns = []
            for _ in range(4):  # fixed number of variations
                z_low = torch.randn(1, model.z_low_dim).to(device)
                logits = model.decode_hierarchy(z_high, z_low)
                pattern = torch.sigmoid(logits).cpu().squeeze(0).numpy()
                genre_patterns.append(pattern)
            patterns_by_genre[genre] = genre_patterns

    return patterns_by_genre


def plot_kl_trends(log_path, save_dir="results/latent_analysis"):
    """
    Plot KL divergence trends across epochs to diagnose posterior collapse.
    Args:
        log_path: Path to training_log.json
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "r") as f:
        history = json.load(f)

    kl_low_train = [ep["kl_low"] for ep in history["train"]]
    kl_high_train = [ep["kl_high"] for ep in history["train"]]
    kl_low_val = [ep["kl_low"] for ep in history["val"]]
    kl_high_val = [ep["kl_high"] for ep in history["val"]]

    epochs = range(len(kl_low_train))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, kl_low_train, label="KL Low (train)", color="blue")
    plt.plot(epochs, kl_high_train, label="KL High (train)", color="red")
    plt.plot(epochs, kl_low_val, "--", label="KL Low (val)", color="blue", alpha=0.6)
    plt.plot(epochs, kl_high_val, "--", label="KL High (val)", color="red", alpha=0.6)
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence Trends (Low vs High level latents)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = save_dir / "kl_trends.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved KL trend plot at {out_path}")
    
    
def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    model.eval()
    all_z_high, all_z_low, all_labels = [], [], []

    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            out = model(patterns)
            mu_low, mu_high = out['mu_low'], out['mu_high']
            all_z_low.append(mu_low.cpu().numpy())
            all_z_high.append(mu_high.cpu().numpy())
            all_labels.append(styles.numpy())

    z_low = np.concatenate(all_z_low, axis=0)
    z_high = np.concatenate(all_z_high, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    fig_high = plot_latent_space_2d(z_high, labels, title="High-level latent space")
    fig_high.savefig("results/latent_analysis/z_high_tsne.png")
    plt.close(fig_high)

    fig_low = plot_latent_space_2d(z_low, labels, title="Low-level latent space")
    fig_low.savefig("results/latent_analysis/z_low_tsne.png")
    plt.close(fig_low)

    return z_high, z_low, labels