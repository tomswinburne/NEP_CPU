#!/usr/bin/env python3
"""
Parse and analyze NEP gradient data from LAMMPS output

The gradient vector contains:
- A_l: dE/dv_l (output layer weights) [num_neurons components]
- B_lj: dE/dw_lj (hidden layer weights) [num_neurons * dim components]
- C_l: dE/db_l (hidden layer biases) [num_neurons components]
- D: dE/db_output (output bias) [1 component]

Total size: num_neurons * (2 + dim) + 1
"""

import numpy as np
import matplotlib.pyplot as plt


class NEPGradientParser:
    def __init__(self, num_neurons, descriptor_dim):
        """
        Initialize parser with NEP model dimensions

        Args:
            num_neurons: Number of hidden layer neurons
            descriptor_dim: Descriptor dimension
        """
        self.num_neurons = num_neurons
        self.dim = descriptor_dim

        # Calculate indices for each gradient component
        self.idx_A_start = 0
        self.idx_A_end = num_neurons

        self.idx_B_start = num_neurons
        self.idx_B_end = num_neurons + num_neurons * descriptor_dim

        self.idx_C_start = num_neurons + num_neurons * descriptor_dim
        self.idx_C_end = 2 * num_neurons + num_neurons * descriptor_dim

        self.idx_D = 2 * num_neurons + num_neurons * descriptor_dim

        self.total_size = num_neurons * (2 + descriptor_dim) + 1

    def load_data(self, filename):
        """
        Load gradient data from LAMMPS output file

        Args:
            filename: Path to gradient.dat file

        Returns:
            data: numpy array of shape (n_timesteps, total_size+1)
                  First column is timestep, rest are gradient components
        """
        data = np.loadtxt(filename)

        # Check if first column is timestep
        if data.shape[1] == self.total_size + 1:
            print(f"Loaded {data.shape[0]} timesteps")
            return data
        elif data.shape[1] == self.total_size:
            print(f"Warning: No timestep column found. Loaded {data.shape[0]} frames")
            # Add fake timestep column
            timesteps = np.arange(data.shape[0]).reshape(-1, 1)
            return np.hstack([timesteps, data])
        else:
            raise ValueError(f"Expected {self.total_size} or {self.total_size+1} columns, "
                           f"got {data.shape[1]}")

    def extract_components(self, data):
        """
        Extract individual gradient components

        Args:
            data: Output from load_data

        Returns:
            dict with keys: 'timestep', 'A_l', 'B_lj', 'C_l', 'D'
        """
        timesteps = data[:, 0]
        gradients = data[:, 1:]  # Skip timestep column

        # Extract A_l (output weights)
        A_l = gradients[:, self.idx_A_start:self.idx_A_end]

        # Extract B_lj (hidden weights) and reshape
        B_lj_flat = gradients[:, self.idx_B_start:self.idx_B_end]
        B_lj = B_lj_flat.reshape(-1, self.num_neurons, self.dim)

        # Extract C_l (hidden biases)
        C_l = gradients[:, self.idx_C_start:self.idx_C_end]

        # Extract D (output bias)
        D = gradients[:, self.idx_D]

        return {
            'timestep': timesteps,
            'A_l': A_l,
            'B_lj': B_lj,
            'C_l': C_l,
            'D': D
        }

    def compute_statistics(self, components):
        """
        Compute useful statistics from gradient components

        Args:
            components: Dict from extract_components

        Returns:
            dict with statistics
        """
        stats = {}

        # Gradient norm per timestep
        A_l = components['A_l']
        B_lj = components['B_lj'].reshape(A_l.shape[0], -1)
        C_l = components['C_l']
        D = components['D'].reshape(-1, 1)

        full_grad = np.hstack([A_l, B_lj, C_l, D])
        stats['gradient_norm'] = np.linalg.norm(full_grad, axis=1)

        # Component-wise norms
        stats['A_norm'] = np.linalg.norm(A_l, axis=1)
        stats['B_norm'] = np.linalg.norm(B_lj, axis=1)
        stats['C_norm'] = np.linalg.norm(C_l, axis=1)
        stats['D_abs'] = np.abs(D.flatten())

        # Mean absolute values
        stats['A_mean'] = np.mean(np.abs(A_l), axis=1)
        stats['B_mean'] = np.mean(np.abs(B_lj), axis=1)
        stats['C_mean'] = np.mean(np.abs(C_l), axis=1)

        return stats

    def plot_gradients(self, components, stats, output_file='gradients.png'):
        """
        Create visualization of gradient evolution

        Args:
            components: Dict from extract_components
            stats: Dict from compute_statistics
            output_file: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        timesteps = components['timestep']

        # Plot 1: Total gradient norm
        axes[0, 0].plot(timesteps, stats['gradient_norm'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Total Gradient Norm')
        axes[0, 0].set_title('Total Gradient Magnitude')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Component norms
        axes[0, 1].plot(timesteps, stats['A_norm'], label='A_l (output weights)', linewidth=2)
        axes[0, 1].plot(timesteps, stats['B_norm'], label='B_lj (hidden weights)', linewidth=2)
        axes[0, 1].plot(timesteps, stats['C_norm'], label='C_l (hidden biases)', linewidth=2)
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Component Norm')
        axes[0, 1].set_title('Gradient Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Heatmap of A_l over time
        im = axes[1, 0].imshow(components['A_l'].T, aspect='auto', cmap='RdBu_r',
                               extent=[timesteps[0], timesteps[-1], 0, self.num_neurons])
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Neuron Index')
        axes[1, 0].set_title('A_l (Output Weight Gradients)')
        plt.colorbar(im, ax=axes[1, 0])

        # Plot 4: Mean absolute values
        axes[1, 1].plot(timesteps, stats['A_mean'], label='A_l', linewidth=2)
        axes[1, 1].plot(timesteps, stats['B_mean'], label='B_lj', linewidth=2)
        axes[1, 1].plot(timesteps, stats['C_mean'], label='C_l', linewidth=2)
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Mean |Gradient|')
        axes[1, 1].set_title('Mean Absolute Gradient Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close()


def main():
    """Example usage"""

    # IMPORTANT: Set these to match your NEP model
    NUM_NEURONS = 30  # Replace with your model's num_neurons1
    DESCRIPTOR_DIM = 50  # Replace with your model's descriptor dimension

    # Initialize parser
    parser = NEPGradientParser(NUM_NEURONS, DESCRIPTOR_DIM)

    # Load data
    print(f"Expected gradient vector size: {parser.total_size}")
    data = parser.load_data('gradient.dat')

    # Extract components
    components = parser.extract_components(data)
    print(f"\nExtracted components:")
    print(f"  A_l shape: {components['A_l'].shape}")
    print(f"  B_lj shape: {components['B_lj'].shape}")
    print(f"  C_l shape: {components['C_l'].shape}")
    print(f"  D shape: {components['D'].shape}")

    # Compute statistics
    stats = parser.compute_statistics(components)
    print(f"\nGradient statistics:")
    print(f"  Mean gradient norm: {np.mean(stats['gradient_norm']):.6e}")
    print(f"  Max gradient norm: {np.max(stats['gradient_norm']):.6e}")
    print(f"  Min gradient norm: {np.min(stats['gradient_norm']):.6e}")

    # Create visualization
    parser.plot_gradients(components, stats)

    # Example: Uncertainty quantification
    # High gradient norm = high uncertainty
    uncertainty = stats['gradient_norm']
    print(f"\nUncertainty metrics:")
    print(f"  Mean uncertainty: {np.mean(uncertainty):.6e}")
    print(f"  Std uncertainty: {np.std(uncertainty):.6e}")

    # Example: Active learning - find most uncertain configurations
    most_uncertain_idx = np.argsort(uncertainty)[-5:]  # Top 5 most uncertain
    print(f"\nMost uncertain timesteps: {components['timestep'][most_uncertain_idx].astype(int)}")

    return components, stats


if __name__ == '__main__':
    main()
