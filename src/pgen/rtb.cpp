//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rt.cpp
//! \brief Problem generator for RT instabilty.
//!
//! Note the gravitational acceleration is hardwired to be 0.1. Density difference is
//! hardwired to be 2.0 in 2D, and is set by the input parameter `problem/rhoh` in 3D
//! (default value is 3.0). This reproduces 2D results of Liska & Wendroff, 3D results of
//! Dimonte et al.
//!
//! FOR 2D HYDRO:
//! Problem domain should be -1/6 < x < 1/6; -0.5 < y < 0.5 with gamma=1.4 to match Liska
//! & Wendroff. Interface is at y=0; perturbation added to Vy. Gravity acts in y-dirn.
//! Special reflecting boundary conditions added in x2 to improve hydrostatic eqm
//! (prevents launching of weak waves) Atwood number A=(d2-d1)/(d2+d1)=1/3. Options:
//!    - iprob = 1  -- Perturb V2 using single mode
//!    - iprob != 1 -- Perturb V2 using multiple mode
//!
//! FOR 3D:
//! Problem domain should be -.05 < x < .05; -.05 < y < .05, -.1 < z < .1, gamma=5/3 to
//! match Dimonte et al.  Interface is at z=0; perturbation added to Vz. Gravity acts in
//! z-dirn. Special reflecting boundary conditions added in x3.  A=1/2.  Options:
//!    - iprob = 1 -- Perturb V3 using single mode
//!    - iprob = 2 -- Perturb V3 using multiple mode
//!    - iprob = 3 -- B rotated by "angle" at interface, multimode perturbation
//!
//! REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)
//========================================================================================

// C headers

// C++ headers
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

void ProjectPressureInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ProjectPressureOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
Real vsq(MeshBlock *pmb, int iout);
// Real cs;


namespace {
// made global to share with BC functions
Real grav_acc;
} // namespace

int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // cs = pin->GetOrAddReal("problem", "cs", 0.1);
  
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);
  if (mesh_size.nx3 == 1) {  // 2D problem
    // Enroll special BCs
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, ProjectPressureInnerX2);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, ProjectPressureOuterX2);
  } else { // 3D problem
    // Enroll special BCs
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, ProjectPressureInnerX3);
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, ProjectPressureOuterX3);
  }
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0,vsq,"vsq");  
  return;
}

// v^2
Real vsq(MeshBlock *pmb, int iout)
{
  Real vsq=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        vsq+= SQR( pmb->phydro->w(IVX,k,j,i) ) + SQR( pmb->phydro->w(IVY,k,j,i) ) + SQR( pmb->phydro->w(IVZ,k,j,i) );
      }
    }
  }

  return vsq;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Rayleigh-Taylor instability test
//========================================================================================


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
// Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  // std::int64_t iseed = -1;
  std::int64_t iseed = -1 - gid;
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;
  // Real press_over_rho = SQR(cs)/(gamma - (gamma/(gamma-1))*SQR(cs));
  Real kx = 2.0*(PI)/(pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min);
  Real ky = 2.0*(PI)/(pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min);
  Real kz = 2.0*(PI)/(pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min);
  
  Real v2;
  
  
  // Read perturbation amplitude, problem switch, density ratio
  Real amp = pin->GetReal("problem","amp");
  int iprob = pin->GetInteger("problem","iprob");
  // Real drat = pin->GetOrAddReal("problem","drat",3.0);


  Real beta_c = pin->GetOrAddReal("problem","beta_c",1.0);
  Real sigma_c = pin->GetOrAddReal("problem","sigma_c",1.0);


  Real press_over_rho_interface = beta_c * sigma_c /2.0;
  Real sigma_h = pin->GetOrAddReal("problem","sigma_h",1.0);
  Real beta_h = press_over_rho_interface/sigma_h * 2.0;


  Real rho_h = 1.0;

  Real Bh = std::sqrt(sigma_h * rho_h);

  // sigma_h/sigma_c = Bh^2/Bc^2 * drat
  // Bh^2/Bc^2 = 1 + (1 - 1/drat)*beta_c 
  // sigma_h/sigma_c  = drat + (drat -1)*beta_c
  Real drat = ( sigma_h/sigma_c+beta_c )/(1.0 + beta_c);

  Real rho_c = rho_h * drat;

  Real Bc = Bh / std::sqrt(1.0 + (1.0 - 1.0/drat)*beta_c);


// press_over_rho = SQR(cs)/(gamma - (gamma/(gamma-1))*SQR(cs));
  Real cs = std::sqrt(press_over_rho_interface * gamma / (1.0 + gamma/(gm1) *press_over_rho_interface) );


// Bc^2 / rho_c = sigma_h * (1 + (1-1/drat)*beta_c)/drat




  // 2D PROBLEM ---------------------------------------------------------------

  if (block_size.nx3 == 1) {
    grav_acc = phydro->hsrc.GetG2();
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real den=1.0;
          if (pcoord->x2v(j) > 0.0) den *= drat;

          if (iprob == 1) {
            v2 = (1.0 + std::cos(kx*pcoord->x1v(i)))*
                                   (1.0 + std::cos(ky*pcoord->x2v(j)))/4.0;
          } else {
           v2 = (ran2(&iseed) - 0.5)*(1.0+std::cos(ky*pcoord->x2v(j)));
          }

          phydro->w(IDN,k,j,i) = den;
          phydro->w(IM1,k,j,i) = 0.0;
          v2 *= amp*cs;
          Real Lorentz = 1.0/std::sqrt(1.0 - SQR(v2));
          phydro->w(IM2,k,j,i) = v2*Lorentz;
          phydro->w(IM3,k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS) {
            phydro->w(IEN,k,j,i) = (press_over_rho_interface*den + grav_acc*den*(pcoord->x2v(j)));
            
          }
        }
      }
    }



    // initialize interface B, same for all iprob
    if (MAGNETIC_FIELDS_ENABLED) {
      // Read magnetic field strength, angle [in degrees, 0 is along +ve X-axis]
      // Real b0 = pin->GetReal("problem","b0");
      Real theta_rot = pin->GetReal("problem","theta_rot");
      theta_rot = (theta_rot/180.)*PI;

      Real L = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
      // Real rotation_region_y_min = 3.0*L/8.0 +  pmy_mesh->mesh_size.x2min;
      // Real rotation_region_y_max = rotation_region_y_min + L/4.0;

      // Real rotation_region_y_min =  L/2.0 + pmy_mesh->mesh_size.x2min;
      // Real rotation_region_y_max = L/2.0 + pmy_mesh->mesh_size.x2min;


      Real rotation_region_y_min = 9.0*L/20.0 +  pmy_mesh->mesh_size.x2min;
      Real rotation_region_y_max = rotation_region_y_min + L/20.0;

      Real Bin = ( Bh * Bc * std::sin(theta_rot) ) / std::sqrt( SQR(Bh) + SQR(Bc) + 2.0*Bh*Bc*std::cos(theta_rot) ) ;
      Real Bhx = Bin;
      Real Bcx = - Bhx;

      Real Bhz = std::sqrt( SQR(Bh) - SQR(Bin) );
      Real Bcz = std::sqrt( SQR(Bc) - SQR(Bin) );

      Real Bx_slope = (Bcx - Bhx) / ( L / 4.0) ; 
      Real Bz_slope = (Bcz - Bhz) / ( L / 4.0) ; 

      Real Bx, Bz;


      
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie+1; i++) {

            if (pcoord->x2v(j) < rotation_region_y_min){
              Bx = Bhx;
              Bz = Bhz;
            }
            else if (pcoord->x2v(j) < L/2.0 + pmy_mesh->mesh_size.x2min){
              Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              Bx = Bx * Bh/B_norm;
              Bz = Bz * Bh/B_norm;
              }
            else if (pcoord->x2v(j) < rotation_region_y_max){
              Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              Bx = Bx * Bc/B_norm;
              Bz = Bz * Bc/B_norm;
            }
            else{
              Bx = Bcx;
              Bz = Bcz;
            }

            pfield->b.x1f(k,j,i) = Bx;
          }
        }
      }
    
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie; i++) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {

            if (pcoord->x2v(j) < rotation_region_y_min){
              Bx = Bhx;
              Bz = Bhz;
            }
            else if (pcoord->x2v(j) < L/2.0 + pmy_mesh->mesh_size.x2min){
              Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              Bx = Bx * Bh/B_norm;
              Bz = Bz * Bh/B_norm;
              }
            else if (pcoord->x2v(j) < rotation_region_y_max){
              Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              Bx = Bx * Bc/B_norm;
              Bz = Bz * Bc/B_norm;
            }
            else{
              Bx = Bcx;
              Bz = Bcz;
            }
            pfield->b.x3f(k,j,i) = Bz;
          }
        }
      }
      if (NON_BAROTROPIC_EOS) {
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
              // phydro->u(IEN,k,j,i) += 0.5*b0*b0;
            }
          }
        }
      }
    }

    // 3D PROBLEM ----------------------------------------------------------------

  } else {
    grav_acc = phydro->hsrc.GetG3();
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real den=1.0;
          if (pcoord->x3v(k) > 0.0) den *= drat;

          Real v3;

          if (iprob == 1) {
            v3 = (1.0 + std::cos(kx*(pcoord->x1v(i))))/8.0
                                   *(1.0 + std::cos(ky*pcoord->x2v(j)))
                                   *(1.0 + std::cos(kz*pcoord->x3v(k)));
          } else {
            v3 = (ran2(&iseed) - 0.5)*(
                1.0 + std::cos(kz*pcoord->x3v(k)));
          }

          phydro->w(IDN,k,j,i) = den;
          phydro->w(IM1,k,j,i) = 0.0;
          phydro->w(IM2,k,j,i) = 0.0;
          v3 *= (amp*cs);

          Real Lorentz = 1.0/std::sqrt(1.0 - SQR(v3));

          phydro->w(IM3,k,j,i) = v3 * Lorentz;
          if (NON_BAROTROPIC_EOS) {
            phydro->w(IPR,k,j,i) =  press_over_rho_interface*den + grav_acc*den*(pcoord->x3v(k));
          }
        }
      }
    }

    // initialize interface B
    if (MAGNETIC_FIELDS_ENABLED) {
      // Read magnetic field strength, angle [in degrees, 0 is along +ve X-axis]
      Real theta_rot = pin->GetReal("problem","theta_rot");
      theta_rot = (theta_rot/180.)*PI;

      Real L = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
      Real rotation_region_z_min = 3.0*L/8.0 +  pmy_mesh->mesh_size.x3min;
      Real rotation_region_z_max = rotation_region_z_min + L/4.0;

      Real Bin = ( Bh * Bc * std::sin(theta_rot) ) / std::sqrt( SQR(Bh) + SQR(Bc) + 2.0*Bh*Bc*std::cos(theta_rot) ) ;
      Real Bhx = Bin;
      Real Bcx = - Bhx;

      Real Bhy = std::sqrt( SQR(Bh) - SQR(Bin) );
      Real Bcy = std::sqrt( SQR(Bc) - SQR(Bin) );

      Real Bx_slope = (Bcx - Bhx) / ( L / 4.0) ; 
      Real By_slope = (Bcy - Bhy) / ( L / 4.0) ; 

      Real Bx, By;
      // angle = (angle/180.)*PI;

      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie+1; i++) {

            if (pcoord->x3v(k) < rotation_region_z_min){
              Bx = Bhx;
              By = Bhy;
            }
            else if (pcoord->x3v(k) < L/2.0 + pmy_mesh->mesh_size.x3min){
              Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bh/B_norm;
              By = By * Bh/B_norm;
              }
            else if (pcoord->x3v(k) < rotation_region_z_max){
              Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bc/B_norm;
              By = By * Bc/B_norm;
            }
            else{
              Bx = Bcx;
              By = Bcy;
            }
   
            pfield->b.x1f(k,j,i) = Bx;
          }
        }
      }
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie; i++) {

            if (pcoord->x3v(k) < rotation_region_z_min){
              Bx = Bhx;
              By = Bhy;
            }
            else if (pcoord->x3v(k) < L/2.0 + pmy_mesh->mesh_size.x3min){
              Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bh/B_norm;
              By = By * Bh/B_norm;
              }
            else if (pcoord->x3v(k) < rotation_region_z_max){
              Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bc/B_norm;
              By = By * Bc/B_norm;
            }
            else{
              Bx = Bcx;
              By = Bcy;
            }
   

            pfield->b.x2f(k,j,i) = By;
          }
        }
      }
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
      if (NON_BAROTROPIC_EOS) {
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
              // phydro->u(IEN,k,j,i) += 0.5*b0*b0;
            }
          }
        }
      }
    }
  } // end of 3D initialization
// Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  } else {
    bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  }

  // Initialize conserved values
  if (MAGNETIC_FIELDS_ENABLED) {
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
        kl, ku);
  } else {
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
    bb.DeleteAthenaArray();
  }
    
  return;
}



//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureInnerX2(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IVY)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVY,k,jl-j,i) = -prim(IVY,k,jl+j-1,i);  // reflect 2-velocity
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,k,jl-j,i) = prim(IPR,k,jl+j-1,i)
                                 - prim(IDN,k,jl+j-1,i)*grav_acc*(2*j-1)*pco->dx2f(j);
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,jl-j,i) = prim(n,k,jl+j-1,i);
          }
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,(jl-j),i) =  b.x1f(k,(jl+j-1),i);
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,(jl-j),i) = -b.x2f(k,(jl+j  ),i);  // reflect 2-field
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,(jl-j),i) =  b.x3f(k,(jl+j-1),i);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureOuterX2(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IVY)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVY,k,ju+j,i) = -prim(IVY,k,ju-j+1,i);  // reflect 2-velocity
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,k,ju+j,i) = prim(IPR,k,ju-j+1,i)
                                 + prim(IDN,k,ju-j+1,i)*grav_acc*(2*j-1)*pco->dx2f(j);
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,ju+j,i) = prim(n,k,ju-j+1,i);
          }
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,(ju+j  ),i) =  b.x1f(k,(ju-j+1),i);
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,(ju+j+1),i) = -b.x2f(k,(ju-j+1),i);  // reflect 2-field
        }
      }
    }

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,(ju+j  ),i) =  b.x3f(k,(ju-j+1),i);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        if (n==(IVZ)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVZ,kl-k,j,i) = -prim(IVZ,kl+k-1,j,i);  // reflect 3-vel
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,kl-k,j,i) = prim(IPR,kl+k-1,j,i)
                                 - prim(IDN,kl+k-1,j,i)*grav_acc*(2*k-1)*pco->dx3f(k);
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,kl-k,j,i) = prim(n,kl+k-1,j,i);
          }
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f((kl-k),j,i) =  b.x1f((kl+k-1),j,i);
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f((kl-k),j,i) =  b.x2f((kl+k-1),j,i);
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f((kl-k),j,i) = -b.x3f((kl+k  ),j,i);  // reflect 3-field
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        if (n==(IVZ)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVZ,ku+k,j,i) = -prim(IVZ,ku-k+1,j,i);  // reflect 3-vel
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,ku+k,j,i) = prim(IPR,ku-k+1,j,i)
                                 + prim(IDN,ku-k+1,j,i)*grav_acc*(2*k-1)*pco->dx3f(k);
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,ku+k,j,i) = prim(n,ku-k+1,j,i);
          }
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f((ku+k  ),j,i) =  b.x1f((ku-k+1),j,i);
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f((ku+k  ),j,i) =  b.x2f((ku-k+1),j,i);
        }
      }
    }

    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f((ku+k+1),j,i) = -b.x3f((ku-k+1),j,i);  // reflect 3-field
        }
      }
    }
  }

  return;
}


// refinement condition: density jump
int RefinementCondition(MeshBlock *pmb) {
  int f2 = pmb->pmy_mesh->f2, f3 = pmb->pmy_mesh->f3;
  AthenaArray<Real> &w = pmb->phydro->w;
  // maximum intercell density ratio
  Real drmax = 1.0;
  for (int k=pmb->ks-f3; k<=pmb->ke+f3; k++) {
    for (int j=pmb->js-f2; j<=pmb->je+f2; j++) {
      for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
        if (w(IDN,k,j,i-1)/w(IDN,k,j,i) > drmax) drmax = w(IDN,k,j,i-1)/w(IDN,k,j,i);
        if (w(IDN,k,j,i+1)/w(IDN,k,j,i) > drmax) drmax = w(IDN,k,j,i+1)/w(IDN,k,j,i);
        if (w(IDN,k,j,i)/w(IDN,k,j,i-1) > drmax) drmax = w(IDN,k,j,i)/w(IDN,k,j,i-1);
        if (w(IDN,k,j,i)/w(IDN,k,j,i+1) > drmax) drmax = w(IDN,k,j,i)/w(IDN,k,j,i+1);
        if (f2) {
          if (w(IDN,k,j-1,i)/w(IDN,k,j,i) > drmax) drmax = w(IDN,k,j-1,i)/w(IDN,k,j,i);
          if (w(IDN,k,j+1,i)/w(IDN,k,j,i) > drmax) drmax = w(IDN,k,j+1,i)/w(IDN,k,j,i);
          if (w(IDN,k,j,i)/w(IDN,k,j-1,i) > drmax) drmax = w(IDN,k,j,i)/w(IDN,k,j-1,i);
          if (w(IDN,k,j,i)/w(IDN,k,j+1,i) > drmax) drmax = w(IDN,k,j,i)/w(IDN,k,j+1,i);
        }
        if (f3) {
          if (w(IDN,k-1,j,i)/w(IDN,k,j,i) > drmax) drmax = w(IDN,k-1,j,i)/w(IDN,k,j,i);
          if (w(IDN,k+1,j,i)/w(IDN,k,j,i) > drmax) drmax = w(IDN,k+1,j,i)/w(IDN,k,j,i);
          if (w(IDN,k,j,i)/w(IDN,k-1,j,i) > drmax) drmax = w(IDN,k,j,i)/w(IDN,k-1,j,i);
          if (w(IDN,k,j,i)/w(IDN,k+1,j,i) > drmax) drmax = w(IDN,k,j,i)/w(IDN,k+1,j,i);
        }
      }
    }
  }
  if (drmax > 1.5) return 1;
  else if (drmax < 1.2) return -1;
  return 0;
}
