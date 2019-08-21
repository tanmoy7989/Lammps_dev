/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(localdensity,PairLOCALDENSITY)

#else

#ifndef LMP_PAIR_LOCALDENSITY_H
#define LMP_PAIR_LOCALDENSITY_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLOCALDENSITY : public Pair {
 public:
  PairLOCALDENSITY(class LAMMPS *);
  virtual ~PairLOCALDENSITY();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  double single(int, int, int, int, double, double, double, double &);	

  virtual int pack_comm(int, int *, double *, int, int *);
  virtual void unpack_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();

 protected:
   double cutmax;

  // data parsed from input file

  int nLD, nrho;
  int **a, **b; // central and neighbor atom filters respectively
  double *uppercut, *lowercut, *uppercutsq, *lowercutsq, *c0, *c2, *c4, *c6;
  double **frho, **rho, *rho_min, *rho_max, *delta_rho;

  // potentials in spline form used for force computation

  double ***frho_spline;

  int nmax;                   // allocated size of per-atom arrays
  double cutforcesq;          // square of global upper cutoff

  // per-atom arrays

  double **localrho;     // local density
  double **fp;           // derivative of embedding function 

  void allocate();     
  void parse_file(char *);
  void file2array(); // UNUSED
  void array2spline();
  void interpolate(int, double, double *, double **);

  // for debugging
  int DEBUG;
  void display();
  void log_data(double, double, double, double); // UNUSED

};

}

#endif
#endif


