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
    Contributing author: Compute average NEP descriptors across all atoms
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(nep/descriptor/avg, ComputeNEPDescriptorAvg)

#else

#ifndef LMP_COMPUTE_NEP_DESCRIPTOR_AVG_H
#define LMP_COMPUTE_NEP_DESCRIPTOR_AVG_H

#include "compute.h"

namespace LAMMPS_NS
{

class ComputeNEPDescriptorAvg : public Compute
{
public:
  ComputeNEPDescriptorAvg(class LAMMPS*, int, char**);
  ~ComputeNEPDescriptorAvg() override;
  void init() override;
  void init_list(int, class NeighList*) override;
  void compute_vector() override;
  double memory_usage() override;

private:
  class NeighList* list;
  class PairNEP* pair_nep;
  int* type_map;     // map from LAMMPS type to NEP element type
};

} // namespace LAMMPS_NS

#endif
#endif
