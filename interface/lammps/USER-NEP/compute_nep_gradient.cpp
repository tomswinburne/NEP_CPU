/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
    Contributing author: Implementation for NEP gradient parameters

    Computes global gradient vector for all NEP neural network parameters:
    - A_l = dE/dv_l = sum_i tanh(sum_j w_lj * D_ij - b_l)
    - B_lj = dE/dw_lj = sum_i v_l * D_ij * (1 - tanh^2(...))
    - C_l = dE/db_l = sum_i -v_l * (1 - tanh^2(...))
    - D = dE/db_output = -N_atoms
------------------------------------------------------------------------- */

#include "compute_nep_gradient.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_NEP.h"
#include "update.h"
#include <cstring>

#define LAMMPS_VERSION_NUMBER 20220324

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeNEPGradient::ComputeNEPGradient(LAMMPS* lmp, int narg, char** arg)
  : Compute(lmp, narg, arg)
{
  if (narg != 3) {
    error->all(FLERR, "Illegal compute nep/gradient command");
  }

  vector_flag = 1;
  extvector = 0;
  pair_nep = nullptr;
  list = nullptr;
  type_map = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeNEPGradient::~ComputeNEPGradient()
{
  memory->destroy(vector);
}

/* ---------------------------------------------------------------------- */

void ComputeNEPGradient::init()
{
  if (force->pair == nullptr) {
    error->all(FLERR, "Compute nep/gradient requires a pair style to be defined");
  }

  // Check if pair style is nep
  pair_nep = dynamic_cast<PairNEP*>(force->pair);
  if (pair_nep == nullptr) {
    error->all(FLERR, "Compute nep/gradient requires pair_style nep");
  }

  // Get dimensions from NEP model
  int num_neurons = pair_nep->nep_model.annmb.num_neurons1;
  int dim = pair_nep->nep_model.annmb.dim;

  if (num_neurons <= 0 || dim <= 0) {
    error->all(FLERR, "NEP model dimensions are invalid");
  }

  // Total size: num_neurons (for A_l/dE/dv_l) + num_neurons * dim (for B_lj/dE/dw_lj)
  //           + num_neurons (for C_l/dE/db_l) + 1 (for dE/db_output)
  size_vector = num_neurons + num_neurons * dim + num_neurons + 1;

  // Allocate vector
  memory->create(vector, size_vector, "nep/gradient:vector");

  // Get type map from pair style
  type_map = pair_nep->type_map;

  // Request a full neighbor list
#if LAMMPS_VERSION_NUMBER >= 20220324
  neighbor->add_request(this, NeighConst::REQ_FULL);
#else
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeNEPGradient::init_list(int /* id */, NeighList* ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeNEPGradient::compute_vector()
{
  invoked_vector = update->ntimestep;

  // Safety check
  if (!vector) {
    error->all(FLERR, "Compute nep/gradient vector not allocated");
  }

  // Initialize vector to zero
  for (int i = 0; i < size_vector; i++) {
    vector[i] = 0.0;
  }

  // Compute gradient using the pair_nep method
  if (pair_nep && list) {
    int nlocal = atom->nlocal;

    // Safety check
    if (nlocal > 0 && (!list->ilist || !list->numneigh || !list->firstneigh)) {
      error->all(FLERR, "Compute nep/gradient neighbor list not properly initialized");
    }

    pair_nep->nep_model.compute_gradient_for_lammps(
      nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh,
      atom->type, type_map, atom->x, vector);
  }

  // Sum across all processors if using MPI
  MPI_Allreduce(MPI_IN_PLACE, vector, size_vector, MPI_DOUBLE, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

double ComputeNEPGradient::memory_usage()
{
  double bytes = 0.0;
  bytes += (double)size_vector * sizeof(double); // vector
  return bytes;
}
