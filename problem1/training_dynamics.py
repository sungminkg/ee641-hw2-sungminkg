"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from metrics import *
from visualize import *

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda'):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Get latent dimension
    z_dim = generator.z_dim
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            d_optimizer.zero_grad()
            
            # Real images
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            
            # Generator wants discriminator to classify fake as real
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # --------- Gradient Norm Logging ---------
            if batch_idx % 10 == 0:
                g_total_norm = torch.sqrt(sum(
                    (p.grad.data.norm(2) ** 2) for p in generator.parameters() if p.grad is not None
                ))
                d_total_norm = torch.sqrt(sum(
                    (p.grad.data.norm(2) ** 2) for p in discriminator.parameters() if p.grad is not None
                ))

                history['g_grad_norm'].append(g_total_norm.item())
                history['d_grad_norm'].append(d_total_norm.item())

            # ----------------------------------------
            
            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            mode_coverage = analyze_mode_coverage(generator, device)
            history['mode_coverage'].append(mode_coverage)
            print(f"Epoch {epoch}: Mode coverage = {mode_coverage:.2f}")
    
    return history


def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
    """
    # TODO: Generate n_samples images
    # Use provided letter classifier to identify generated letters
    # Count unique letters produced
    # Return coverage score (0 to 1)
    generator.eval()
    z_dim = generator.z_dim
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        fake_images = generator(z)

    from metrics import mode_coverage_score
    result = mode_coverage_score(fake_images)
    coverage = result['coverage_score']

    generator.train()
    return coverage


def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    # TODO: Plot mode coverage over time
    # Show which letters survive and which disappear
    from visualize import plot_training_history

    fig = plot_training_history(history, save_path=save_path)
    return fig