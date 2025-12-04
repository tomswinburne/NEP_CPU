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
    Contributing author: Thomas D Swinburne, tomswinburne.github.io
------------------------------------------------------------------------- */

#include "compute_nep_atom.h"
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

ComputeNEPAtom::ComputeNEPAtom(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg)
{
  if (narg != 3) {
    error->all(FLERR, "Illegal compute nep/atom command");
  }

  peratom_flag = 1;
  nmax = 0;
  ndims = 0;
  pair_nep = nullptr;
  list = nullptr;
  type_map = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeNEPAtom::~ComputeNEPAtom()
{
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeNEPAtom::init()
{
  if (force->pair == nullptr) {
    error->all(FLERR, "Compute nep/atom requires a pair style to be defined");
  }

  // Check if pair style is nep
  pair_nep = dynamic_cast<PairNEP*>(force->pair);
  if (pair_nep == nullptr) {
    error->all(FLERR, "Compute nep/atom requires pair_style nep");
  }

  // Get descriptor dimension from NEP model
  ndims = pair_nep->nep_model.annmb.dim;
  if (ndims <= 0) {
    error->all(FLERR, "NEP model descriptor dimension is invalid");
  }

  // Set the size of the per-atom array
  size_peratom_cols = ndims;

  // Get cutoff from pair style
  cutsq = pair_nep->cutoffsq;

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

void ComputeNEPAtom::init_list(int /* id */, NeighList* ptr) { list = ptr; }

/* ---------------------------------------------------------------------- */

void ComputeNEPAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // Grow array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(array_atom);
    nmax = atom->nmax;
    memory->create(array_atom, nmax, ndims, "nep/atom:array_atom");
  }

  // Initialize array to zero
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < ndims; j++) {
      array_atom[i][j] = 0.0;
    }
  }

  // Compute descriptors using the pair_nep method
  if (pair_nep && list) {
    pair_nep->nep_model.compute_descriptors_for_lammps(
      nlocal, list->inum, list->ilist, list->numneigh, list->firstneigh, atom->type, type_map,
      atom->x, array_atom);
  }
}

/* ---------------------------------------------------------------------- */

double ComputeNEPAtom::memory_usage()
{
  double bytes = 0.0;
  bytes += (double)nmax * ndims * sizeof(double); // array_atom
  return bytes;
}
