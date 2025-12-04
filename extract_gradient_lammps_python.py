#!/usr/bin/env python3
"""
Safe extraction of NEP gradients from LAMMPS Python interface

Usage:
    from extract_gradient_lammps_python import extract_nep_gradient

    gradient = extract_nep_gradient(lmp, "G")  # where "G" is compute ID
"""

import numpy as np
from lammps import lammps, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_SIZE_VECTOR


def extract_nep_gradient(lmp, compute_id):
    """
    Safely extract NEP gradient from LAMMPS compute

    Args:
        lmp: LAMMPS object
        compute_id: ID of the compute nep/gradient (e.g., "G")

    Returns:
        numpy array with gradient values
    """

    # Step 1: Force compute to run (critical!)
    # The compute might not have been invoked yet
    lmp.command("run 0 post no")

    # Step 2: Get the size of the vector
    try:
        size = lmp.extract_compute(compute_id, LMP_STYLE_GLOBAL, LMP_SIZE_VECTOR)
        print(f"Gradient vector size: {size}")
    except Exception as e:
        print(f"Error getting vector size: {e}")
        print("Trying alternative method...")
        # Alternative: run one step and try again
        lmp.command("run 1 post no")
        size = lmp.extract_compute(compute_id, LMP_STYLE_GLOBAL, LMP_SIZE_VECTOR)

    # Step 3: Extract the vector
    try:
        # Use numpy interface if available (LAMMPS 29 Sep 2021 or later)
        if hasattr(lmp, 'numpy'):
            gradient = lmp.numpy.extract_compute(compute_id, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            if gradient is not None:
                return gradient.copy()  # IMPORTANT: copy to avoid dangling pointer
            else:
                raise ValueError("Compute returned None")
        else:
            # Fallback for older LAMMPS versions
            gradient_ptr = lmp.extract_compute(compute_id, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            if gradient_ptr is None:
                raise ValueError("Compute returned None pointer")

            # Convert ctypes pointer to numpy array
            import ctypes
            gradient = np.ctypeslib.as_array(
                (ctypes.c_double * size).from_address(int(gradient_ptr))
            )
            return gradient.copy()  # IMPORTANT: copy to avoid dangling pointer

    except Exception as e:
        print(f"Error extracting gradient: {e}")
        print("\nDebugging info:")
        print(f"  Compute ID: {compute_id}")
        print(f"  Expected size: {size}")

        # Try to get more info about the compute
        try:
            info = lmp.extract_compute(compute_id, 0, 0)  # Get compute pointer
            print(f"  Compute pointer: {info}")
        except:
            pass

        raise


def parse_gradient_components(gradient, num_neurons, descriptor_dim):
    """
    Parse gradient vector into components

    Args:
        gradient: numpy array from extract_nep_gradient
        num_neurons: number of hidden neurons
        descriptor_dim: descriptor dimension

    Returns:
        dict with 'A_l', 'B_lj', 'C_l', 'D'
    """
    expected_size = num_neurons * (2 + descriptor_dim) + 1

    if len(gradient) != expected_size:
        raise ValueError(f"Gradient size mismatch: got {len(gradient)}, expected {expected_size}")

    idx = 0

    # Extract A_l
    A_l = gradient[idx:idx + num_neurons]
    idx += num_neurons

    # Extract B_lj
    B_lj_flat = gradient[idx:idx + num_neurons * descriptor_dim]
    B_lj = B_lj_flat.reshape(num_neurons, descriptor_dim)
    idx += num_neurons * descriptor_dim

    # Extract C_l
    C_l = gradient[idx:idx + num_neurons]
    idx += num_neurons

    # Extract D
    D = gradient[idx]

    return {
        'A_l': A_l,
        'B_lj': B_lj,
        'C_l': C_l,
        'D': D
    }


def example_usage():
    """Example of how to use these functions"""

    from lammps import lammps

    # Initialize LAMMPS
    lmp = lammps()

    # Run your LAMMPS script
    lmp.file("test_nep.in")

    # Extract gradient
    try:
        gradient = extract_nep_gradient(lmp, "grad")  # "grad" is the compute ID
        print(f"Successfully extracted gradient: shape = {gradient.shape}")

        # Parse into components (replace with your model dimensions)
        num_neurons = 30
        descriptor_dim = 50

        components = parse_gradient_components(gradient, num_neurons, descriptor_dim)

        print(f"\nGradient components:")
        print(f"  A_l: {components['A_l'].shape}")
        print(f"  B_lj: {components['B_lj'].shape}")
        print(f"  C_l: {components['C_l'].shape}")
        print(f"  D: {components['D']}")

        # Compute gradient norm (uncertainty metric)
        grad_norm = np.linalg.norm(gradient)
        print(f"\nGradient norm: {grad_norm:.6e}")

    except Exception as e:
        print(f"Failed to extract gradient: {e}")
        import traceback
        traceback.print_exc()

    finally:
        lmp.close()


if __name__ == '__main__':
    print("NEP Gradient Extraction Utility")
    print("=" * 50)
    example_usage()
