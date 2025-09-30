"""
Main training script for hierarchical VAE experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import numpy as np

from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import kl_annealing_schedule, temperature_annealing_schedule
from metrics import drum_pattern_validity, sequence_diversity
from visualize import *


def compute_hierarchical_elbo(recon_x, x, mu_low, logvar_low, mu_high, logvar_high, beta=1.0):
    """
    Compute Evidence Lower Bound (ELBO) for hierarchical VAE.
    
    ELBO = E[log p(x|z_low)] - beta * KL(q(z_low|x) || p(z_low|z_high)) 
           - beta * KL(q(z_high|z_low) || p(z_high))
    
    Args:
        recon_x: Reconstructed pattern logits [batch, 16, 9]
        x: Original patterns [batch, 16, 9]
        mu_low, logvar_low: Low-level latent parameters
        mu_high, logvar_high: High-level latent parameters
        beta: KL weight for beta-VAE
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction component
        kl_low: KL divergence for low-level latent
        kl_high: KL divergence for high-level latent
    """
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_x.reshape(-1), x.reshape(-1), reduction='sum'
    )
    
    # KL divergence
    kl_high = -0.5 * torch.sum(1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
    kl_low = -0.5 * torch.sum(1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
    
    # Apply Free bits trick for stable optimization
    free_bits = 0.1
    kl_high = torch.sum(torch.clamp(kl_high, min=free_bits))
    kl_low  = torch.sum(torch.clamp(kl_low,  min=free_bits))
    
    # Total loss
    total_loss = recon_loss + beta * (kl_low + kl_high)
    
    return total_loss, recon_loss, kl_low, kl_high

def train_epoch(model, data_loader, optimizer, epoch, device, config):
    """
    Train model for one epoch with annealing schedules.
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    # Metrics
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kl_low': 0.0,
        'kl_high': 0.0
    }
    
    # Get annealing parameters for this epoch
    beta = kl_annealing_schedule(epoch, method=config['kl_anneal_method'])
    temperature = temperature_annealing_schedule(epoch)
    
    for batch_idx, (patterns, styles, densities) in enumerate(data_loader):
        patterns = patterns.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(patterns, beta=beta)
        recon = out['recon']
        mu_low, logvar_low = out['mu_low'], out['logvar_low']
        mu_high, logvar_high = out['mu_high'], out['logvar_high']
        
        # Compute loss
        loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
            recon, patterns, mu_low, logvar_low, mu_high, logvar_high, beta
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics['total_loss'] += loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['kl_low'] += kl_low.item()
        metrics['kl_high'] += kl_high.item()
        
    # Logging
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, "
                f"Loss={loss.item()/len(patterns):.4f}, "
                f"Recon={recon_loss.item()/len(patterns):.4f}, "
                f"KL_low={kl_low.item()/len(patterns):.4f}, "
                f"KL_high={kl_high.item()/len(patterns):.4f}, "
                f"Beta={beta:.3f}, Temp={temperature:.2f}")
    
    n_batches = len(data_loader)
    return {
        'loss': metrics['total_loss'] / n_batches,
        'recon_loss': metrics['recon_loss'] / n_batches,
        'kl_low': metrics['kl_low'] / n_batches,
        'kl_high': metrics['kl_high'] / n_batches,
        'beta': beta,
        'temperature': temperature
    }

def main():
    """
    Main training entry point for hierarchical VAE experiments.
    """
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'z_high_dim': 4,
        'z_low_dim': 12,
        'kl_anneal_method': 'linear',  # 'linear', 'cyclical', or 'sigmoid'
        'data_dir': '../data/drums',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = DrumPatternDataset(config['data_dir'], split='train')
    val_dataset = DrumPatternDataset(config['data_dir'], split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model and optimizer
    model = HierarchicalDrumVAE(
        z_high_dim=config['z_high_dim'],
        z_low_dim=config['z_low_dim']
    ).to(config['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'config': config
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, 
            config['device'], config
        )
        history['train'].append(train_metrics)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_metrics = {
                'total_loss': 0,
                'recon_loss': 0,
                'kl_low': 0,
                'kl_high': 0,
                'validity': 0,
                'diversity': 0
            }
            
            all_patterns = []

            with torch.no_grad():
                for patterns, styles, densities in val_loader:
                    patterns = patterns.to(config['device'])
                    out = model(patterns)
                    recon = out['recon']
                    mu_low, logvar_low = out['mu_low'], out['logvar_low']
                    mu_high, logvar_high = out['mu_high'], out['logvar_high']
                    
                    loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
                        recon, patterns, mu_low, logvar_low, mu_high, logvar_high
                    )

                    val_metrics['total_loss'] += loss.item()
                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['kl_low'] += kl_low.item()
                    val_metrics['kl_high'] += kl_high.item()

                    sampled = (torch.sigmoid(recon) > 0.5).float().cpu()
                    all_patterns.append(sampled)

            all_patterns = torch.cat(all_patterns, dim=0)

            val_metrics['validity'] = drum_pattern_validity(all_patterns)
            val_metrics['diversity'] = sequence_diversity(all_patterns)

            # Average validation metrics
            n_val = len(val_dataset)
            for key in ['total_loss', 'recon_loss', 'kl_low', 'kl_high']:
                val_metrics[key] /= n_val
            
            history['val'].append(val_metrics)
            
            print(f"Epoch {epoch} Validation - "
                  f"Loss: {val_metrics['total_loss']:.4f} "
                  f"KL_high: {val_metrics['kl_high']:.4f} "
                  f"KL_low: {val_metrics['kl_low']:.4f} "
                  f"Validity: {val_metrics['validity']:.3f} "
                  f"Diversity: {val_metrics['diversity']:.3f}")

        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")

    # Save final model and history
    torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")
    
    def to_serializable(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object {obj} not serializable")

    with open(f"{config['results_dir']}/training_log.json", "w") as f:
        json.dump(history, f, indent=2, default=to_serializable)
    
    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()