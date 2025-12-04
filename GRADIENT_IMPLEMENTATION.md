# NEP Gradient Parameters Implementation

## Overview

This implementation computes the gradient of the NEP energy with respect to neural network parameters as a global vector. This is useful for uncertainty quantification, active learning, and sensitivity analysis.

## Mathematical Formulation

The NEP energy reads:

```
E = Σ_i Σ_l v_l tanh(Σ_j w_lj D_ij - b_l) - b_output
```

where:
- `D_ij` is the descriptor vector for atom `i` (with `j` indexing the descriptor components)
- `w_lj` are the weights of the hidden layer (first layer)
- `b_l` are the biases of the hidden layer
- `v_l` are the weights of the output layer (second layer)
- `b_output` is the output layer bias
- `l` indexes the hidden neurons
- `i` indexes the atoms

### Perturbation Analysis

Consider perturbations `v → v + δv`, `w → w + δw`, `b → b + δb`:

```
E → E + dE
dE = Σ_l δv_l A_l + Σ_lj δw_lj B_lj + Σ_l δb_l C_l + δb_output D
```

where:

**A_l** (gradient w.r.t. output layer weights `v_l`):
```
A_l = ∂E/∂v_l = Σ_i tanh(Σ_j w_lj D_ij - b_l)
```

**B_lj** (gradient w.r.t. hidden layer weights `w_lj`):
```
B_lj = ∂E/∂w_lj = Σ_i v_l D_ij (1 - [tanh(Σ_j w_lj D_ij - b_l)]²)
```

**C_l** (gradient w.r.t. hidden layer biases `b_l`):
```
C_l = ∂E/∂b_l = Σ_i (-v_l) (1 - [tanh(Σ_j w_lj D_ij - b_l)]²)
```

**D** (gradient w.r.t. output layer bias `b_output`):
```
D = ∂E/∂b_output = -N_atoms
```

## Implementation

### C++ Core Functions

#### 1. `NEP3::find_gradient_parameters`
**Location**: [src/nep.cpp:3184](src/nep.cpp#L3184)

Computes the global gradient vector containing both A_l and B_lj components.

**Interface**:
```cpp
void NEP3::find_gradient_parameters(
  const std::vector<int>& type,
  const std::vector<double>& box,
  const std::vector<double>& position,
  std::vector<double>& gradient_vector
);
```

**Output format**:
The `gradient_vector` has size `num_neurons * (1 + dim) + num_neurons + 1` with layout:
- `[0 : num_neurons-1]` → A_l components (∂E/∂v_l)
- `[num_neurons : num_neurons + num_neurons*dim - 1]` → B_lj components (∂E/∂w_lj), flattened as `l * dim + j`
- `[num_neurons + num_neurons*dim : 2*num_neurons + num_neurons*dim - 1]` → C_l components (∂E/∂b_l)
- `[2*num_neurons + num_neurons*dim]` → D component (∂E/∂b_output)

#### 2. `NEP3::compute_gradient_for_lammps`
**Location**: [src/nep.cpp:3571](src/nep.cpp#L3571)

LAMMPS interface wrapper that converts LAMMPS data structures to the format required by `find_gradient_parameters`.

### LAMMPS Interface

#### Compute Style: `nep/gradient`
**Files**:
- Header: [interface/lammps/USER-NEP/compute_nep_gradient.h](interface/lammps/USER-NEP/compute_nep_gradient.h)
- Implementation: [interface/lammps/USER-NEP/compute_nep_gradient.cpp](interface/lammps/USER-NEP/compute_nep_gradient.cpp)

**Usage in LAMMPS**:
```lammps
compute grad all nep/gradient
```

This creates a global vector compute that can be accessed as `c_grad[i]` where `i` ranges from 1 to the total size of the gradient vector.

**Output**:
- Returns a global vector (not per-atom)
- Automatically handles MPI communication (sums across all processors)
- Vector size: `num_neurons * (1 + dim) + num_neurons + 1`
  - `num_neurons` for A_l (output weights)
  - `num_neurons * dim` for B_lj (hidden weights)
  - `num_neurons` for C_l (hidden biases)
  - `1` for D (output bias)

## Usage Example

### LAMMPS Input Script

```lammps
# Set up NEP potential
pair_style      nep
pair_coeff      * * nep.txt C

# Compute gradient vector
compute         grad all nep/gradient

# Option 1: Print gradient to file every 100 steps
variable        grad1 equal c_grad[1]
variable        grad2 equal c_grad[2]
fix             gradout all print 100 "${grad1} ${grad2} ..." file gradient.dat

# Option 2: Use in thermo output (for small gradient vectors)
thermo_style    custom step pe c_grad[1] c_grad[2] c_grad[3]

# Run simulation
run             1000
```

### Python Post-processing

```python
import numpy as np

# Load gradient data
data = np.loadtxt('gradient.dat')

# Extract gradient components
num_neurons = 30  # Example: replace with your model's value
dim = 50          # Example: replace with your descriptor dimension

# Split the gradient vector
A_l = data[:, :num_neurons]                                          # ∂E/∂v_l
B_lj = data[:, num_neurons:num_neurons + num_neurons*dim]           # ∂E/∂w_lj
B_lj = B_lj.reshape(-1, num_neurons, dim)
C_l = data[:, num_neurons + num_neurons*dim:2*num_neurons + num_neurons*dim]  # ∂E/∂b_l
D = data[:, -1]                                                       # ∂E/∂b_output

# Compute uncertainty estimate (example)
gradient_norm = np.linalg.norm(data, axis=1)
```

## Compilation

### Adding to LAMMPS Build

When copying the NEP interface to LAMMPS, make sure to include the new files:

```bash
cd ${NEP_CPU_PATH}
cp src/* interface/lammps/USER-NEP/
cp -r interface/lammps/USER-NEP ${LAMMPS_PATH}/src/
```

The new files that need to be included:
- `compute_nep_gradient.h`
- `compute_nep_gradient.cpp`
- `compute_nep_descriptor_avg.h` (optional - for averaging descriptors)
- `compute_nep_descriptor_avg.cpp` (optional - for averaging descriptors)

### Make-based Build

```bash
cd ${LAMMPS_PATH}/src
make yes-USER-NEP
make serial  # or your target
```

### CMake-based Build

```bash
cd ${LAMMPS_PATH}/build
cmake -D PKG_USER-NEP=on ../cmake
cmake --build .
```

## Technical Details

### Memory Layout

The gradient vector is laid out in memory as:
```
[A_0, ..., A_{N-1}, B_{0,0}, ..., B_{0,D-1}, B_{1,0}, ..., B_{N-1,D-1}, C_0, ..., C_{N-1}, D]
```

where:
- `N = num_neurons`
- `D = dim` (descriptor dimension)
- `A_l` = ∂E/∂v_l (output layer weights)
- `B_lj` = ∂E/∂w_lj (hidden layer weights)
- `C_l` = ∂E/∂b_l (hidden layer biases)
- `D` = ∂E/∂b_output (output layer bias)
- Total size = `N + N*D + N + 1 = N*(2+D) + 1`

### Computational Complexity

- Time complexity: O(N_atoms × N_neighbors × D)
- Memory usage: O(N_atoms × N_neurons × D)
- Same order as standard NEP force computation

### MPI Parallelization

The implementation correctly handles MPI parallelization:
1. Each processor computes gradients for its local atoms
2. `MPI_Allreduce` sums contributions across all processors
3. Result is identical on all processors (global vector)

## Validation

To validate the implementation, you can check:

1. **Energy gradient consistency**: Numerical derivative of energy with respect to parameters should match the computed gradients

2. **Conservation**: The sum `Σ_i` should be properly accumulated

3. **Dimensional analysis**:
   - A_l has units of [dimensionless] (sum of tanh outputs)
   - B_lj has units of [descriptor units] (involves D_ij)
   - C_l has units of [dimensionless] (involves only tanh derivatives)
   - D has units of [dimensionless] (just a count of atoms)

## Applications

This implementation enables:

1. **Uncertainty Quantification**: Gradient magnitude indicates model sensitivity
2. **Active Learning**: Select configurations with high gradient norms for training
3. **Parameter Sensitivity**: Identify which parameters most affect predictions
4. **Ensemble Methods**: Weight predictions by gradient-based confidence
5. **Bayesian Inference**: Use gradients for Hamiltonian Monte Carlo sampling

## References

- Fan et al., "GPUMD: A package for constructing accurate machine-learned potentials", J. Chem. Phys. 157, 114801 (2022)
- NEP_CPU Repository: https://github.com/brucefan1983/NEP_CPU

## Notes

- The implementation reuses the existing `find_B_projection` infrastructure
- The `B_projection` array already contains per-atom gradients; this implementation simply aggregates them
- For multi-element systems, the type mapping is handled automatically through the existing NEP infrastructure
