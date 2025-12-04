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
    Contributing author: Compute average NEP descriptors

    Computes the average descriptor vector across all atoms as a global vector.
    This allows using it with compute reduce and other global operations.
------------------------------------------------------------------------- */

#include "compute_nep_descriptor_avg.h"
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

ComputeNEPDescriptorAvg::ComputeNEPDescriptorAvg(LAMMPS* lmp, int narg, char** arg)
  : Compute(lmp, narg, arg)
{
  if (narg != 3) {
    error->all(FLERR, "Illegal compute nep/descriptor/avg command");
  }

  vector_flag = 1;
  extvector = 0;
  pair_nep = nullptr;
  list = nullptr;
  type_map = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeNEPDescriptorAvg::~ComputeNEPDescriptorAvg()
{
  memory->destroy(vector);
}

/* ---------------------------------------------------------------------- */

void ComputeNEPDescriptorAvg::init()
{
  if (force->pair == nullptr) {
    error->all(FLERR, "Compute nep/descriptor/avg requires a pair style to be defined");
  }

  // Check if pair style is nep
  pair_nep = dynamic_cast<PairNEP*>(force->pair);
  if (pair_nep == nullptr) {
    error->all(FLERR, "Compute nep/descriptor/avg requires pair_style nep");
  }

  // Get descriptor dimension from NEP model
  int ndims = pair_nep->nep_model.annmb.dim;
  if (ndims <= 0) {
    error->all(FLERR, "NEP model descriptor dimension is invalid");
  }

  // Set size as descriptor dimension
  size_vector = ndims;

  // Allocate vector
  memory->create(vector, size_vector, "nep/descriptor/avg:vector");

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

void ComputeNEPDescriptorAvg::init_list(int /* id */, NeighList* ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeNEPDescriptorAvg::compute_vector()
{
  invoked_vector = update->ntimestep;

  int nlocal = atom->nlocal;
  int ndims = size_vector;

  // Initialize vector to zero
  for (int i = 0; i < ndims; i++) {
    vector[i] = 0.0;
  }

  // Allocate temporary array for per-atom descriptors
  double** descriptors = nullptr;
  memory->create(descriptors, nlocal, ndims, "nep/descriptor/avg:descriptors");

  // Compute descriptors for all local atoms
  if (pair_nep && list) {
    pair_nep->nep_model.compute_descriptors_for_lammps(
      nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh,
      atom->type, type_map, atom->x, descriptors);
  }

  // Sum descriptors across all local atoms
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < ndims; j++) {
      vector[j] += descriptors[i][j];
    }
  }

  // Clean up temporary array
  memory->destroy(descriptors);

  // Sum across all processors and get total atom count
  double local_natoms = static_cast<double>(nlocal);
  double total_natoms;

  MPI_Allreduce(&local_natoms, &total_natoms, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, vector, ndims, MPI_DOUBLE, MPI_SUM, world);

  // Divide by total number of atoms to get average
  if (total_natoms > 0.0) {
    for (int i = 0; i < ndims; i++) {
      vector[i] /= total_natoms;
    }
  }
}

/* ---------------------------------------------------------------------- */

double ComputeNEPDescriptorAvg::memory_usage()
{
  double bytes = 0.0;
  bytes += (double)size_vector * sizeof(double); // vector
  return bytes;
}
