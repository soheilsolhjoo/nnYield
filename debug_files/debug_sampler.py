import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.trainer import Trainer
from src.utils import load_config

def test_sampler_randomness():
    print("--- Testing Dynamic Sampler Randomness ---")
    config = load_config("configs/von_mises_synthetic.yaml")
    trainer = Trainer(config)
    
    # Run 1
    points_1 = trainer._sample_dynamic_surface(1000).numpy()
    # Run 2
    points_2 = trainer._sample_dynamic_surface(1000).numpy()
    
    # Check for exact duplicates (Caching Bug)
    if np.allclose(points_1, points_2):
        print("!!! CRITICAL FAIL: Sampler is returning identical data (Caching Issue) !!!")
    else:
        print("SUCCESS: Sampler is generating unique data every step.")
        
    # Visualize distribution
    fig = plt.figure(figsize=(12, 6))
    
    # Plot 1: Run 1 (Blue)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_1[:,0], points_1[:,1], points_1[:,2], c='b', s=2, alpha=0.5)
    ax1.set_title("Run 1 (Random Gaussian)")
    
    # Plot 2: Run 2 (Red)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_2[:,0], points_2[:,1], points_2[:,2], c='r', s=2, alpha=0.5)
    ax2.set_title("Run 2 (Should be different)")
    
    plt.savefig("debug_sampler_randomness.png")
    print("Saved visualization to 'debug_sampler_randomness.png'")

if __name__ == "__main__":
    test_sampler_randomness()