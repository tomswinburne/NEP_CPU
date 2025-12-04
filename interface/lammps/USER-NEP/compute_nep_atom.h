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

#ifdef COMPUTE_CLASS

ComputeStyle(nep/atom, ComputeNEPAtom)

#else

#ifndef LMP_COMPUTE_NEP_ATOM_H
#define LMP_COMPUTE_NEP_ATOM_H

#include "compute.h"

namespace LAMMPS_NS
{

class ComputeNEPAtom : public Compute
{
public:
  ComputeNEPAtom(class LAMMPS*, int, char**);
  ~ComputeNEPAtom() override;
  void init() override;
  void init_list(int, class NeighList*) override;
  void compute_peratom() override;
  double memory_usage() override;

private:
  int nmax;          // allocated size of per-atom arrays
  int ndims;         // number of descriptor dimensions
  double cutsq;      // cutoff squared
  class NeighList* list;
  class PairNEP* pair_nep;
  int* type_map;     // map from LAMMPS type to NEP element type
};

} // namespace LAMMPS_NS

#endif
#endif
