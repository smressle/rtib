//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file constant_acc.cpp
//! \brief source terms due to constant acceleration (e.g. for RT instability)

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../hydro.hpp"
#include "hydro_srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn void HydroSourceTerms::ConstantAcceleration
//! \brief Adds source terms for constant acceleration to conserved variables

void HydroSourceTerms::ConstantAcceleration(const Real dt,const AthenaArray<Real> *flux,
                                            const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                                            AthenaArray<Real> &cons) {
  MeshBlock *pmb = pmy_hydro_->pmy_block;
  Real gam = *pmb->peos->GetGamma();

  // acceleration in 1-direction
  if (g1_!=0.0) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {

          Real inertial_term = prim(IDN,k,j,i);

          if (RELATIVISTIC_DYNAMICS==1){
            Real bsq = SQR(bcc(IB1,k,j,i)) + SQR(bcc(IB2,k,j,i)) + SQR(bcc(IB3,k,j,i)) ;
            inertial_term = prim(IDN,k,j,i)*(1.0 + gam/(gam-1.0)*prim(IPR,k,j,i)/prim(IDN,k,j,i) ) + bsq;
          }
          Real src = dt*inertial_term*g1_;
          cons(IM1,k,j,i) += src;
          if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVX,k,j,i);
        }
      }
    }
  }

  // acceleration in 2-direction
  if (g2_!=0.0) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real inertial_term = prim(IDN,k,j,i);

          if (RELATIVISTIC_DYNAMICS==1){
            Real bsq = SQR(bcc(IB1,k,j,i)) + SQR(bcc(IB2,k,j,i)) + SQR(bcc(IB3,k,j,i)) ;
            inertial_term = prim(IDN,k,j,i)*(1.0 + gam/(gam-1.0)*prim(IPR,k,j,i)/prim(IDN,k,j,i) ) + bsq;
          }
          Real src = dt*inertial_term*g2_;
          cons(IM2,k,j,i) += src;
          if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVY,k,j,i);
        }
      }
    }
  }

  // acceleration in 3-direction
  if (g3_!=0.0) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real inertial_term = prim(IDN,k,j,i);

          if (RELATIVISTIC_DYNAMICS==1){
            Real bsq = SQR(bcc(IB1,k,j,i)) + SQR(bcc(IB2,k,j,i)) + SQR(bcc(IB3,k,j,i)) ;
            inertial_term = prim(IDN,k,j,i)*(1.0 + gam/(gam-1.0)*prim(IPR,k,j,i)/prim(IDN,k,j,i) ) + bsq;
          }
          Real src = dt*inertial_term*g3_;
          cons(IM3,k,j,i) += src;
          if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVZ,k,j,i);
        }
      }
    }
  }

  return;
}
