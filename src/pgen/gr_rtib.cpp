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

void linear_metric_2D(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
void linear_metric_3D(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);


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


    if (mesh_size.nx3>1) EnrollUserMetric(linear_metric_3D);
    else EnrollUserMetric(linear_metric_2D);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate space for scratch arrays
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie + NGHOST + 1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie + NGHOST + 1);

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


  // Prepare scratch arrays
  AthenaArray<Real> &g = ruser_meshblock_data[0];
  AthenaArray<Real> &gi = ruser_meshblock_data[1];

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

  Real L;
  if (block_size.nx3==1) L = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  else L = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
  Real length_of_rotation_region = pin->GetOrAddReal("problem","length_of_rotation_region",L/10.0);

// Bc^2 / rho_c = sigma_h * (1 + (1-1/drat)*beta_c)/drat




  // 2D PROBLEM ---------------------------------------------------------------

  if (block_size.nx3 == 1) {
    grav_acc = pin->GetReal("problem", "grav_acc");
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pcoord->CellMetric(k, j, il, iu, g, gi);
        for (int i=is; i<=ie; i++) {
          Real dh = 1.0;
          Real dc = dh * drat;
          Real den=1.0;
          if (pcoord->x2v(j) > 0.0) den *= drat;

          Real exp_arg_term,press,Bmag;
          if (pcoord->x2v(j) > 0.0){ // cold
            exp_arg_term = grav_acc / sigma_c * (2.0 + gamma/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
            press = press_over_rho_interface*dc * std::exp(pcoord->x2v(j)*exp_arg_term);
            den = dc * std::exp(pcoord->x2v(j)*exp_arg_term);
            Bmag = Bc * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

          }
          else{ // hot
            exp_arg_term = grav_acc / sigma_h * (2.0 + gamma/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
            press = press_over_rho_interface*dh * std::exp(pcoord->x2v(j)*exp_arg_term);
            den = dh * std::exp(pcoord->x2v(j)*exp_arg_term);
            Bmag = Bh * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));
          }



          if (iprob == 1) {
            v2 = (1.0 + std::cos(kx*pcoord->x1v(i)))*
                                   (1.0 + std::cos(ky*pcoord->x2v(j)))/4.0;
          } else {
           v2 = (ran2(&iseed) - 0.5)*(1.0+std::cos(ky*pcoord->x2v(j)));
          }

          phydro->w(IDN,k,j,i) = den;
          phydro->w(IM1,k,j,i) = 0.0;
          v2 *= amp*cs;
          Real Lorentz = 1.0/std::sqrt(-g(I00) - SQR(v2));
          Real uu0 = Lorentz;
          Real uu1 = 0.0;
          Real uu2 = v2*Lorentz;
          Real uu3 = 0.0;

          Real uu1_ = uu1 - gi(I01,i) / gi(I00,i) * uu0;
          Real uu2_ = uu2 - gi(I02,i) / gi(I00,i) * uu0;
          Real uu3_ = uu3 - gi(I03,i) / gi(I00,i) * uu0;
          phydro->w(IM2,k,j,i) = uu2_;
          phydro->w(IM3,k,j,i) = 0.0;
          if (NON_BAROTROPIC_EOS) {
            phydro->w(IEN,k,j,i) = press;
            // phydro->w(IEN,k,j,i) = (press_over_rho_interface*den + grav_acc*den*(pcoord->x2v(j)));
            
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

      // Real L = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
      // Real rotation_region_y_min = 3.0*L/8.0 +  pmy_mesh->mesh_size.x2min;
      // Real rotation_region_y_max = rotation_region_y_min + L/4.0;

      // Real rotation_region_y_min =  L/2.0 + pmy_mesh->mesh_size.x2min;
      // Real rotation_region_y_max = L/2.0 + pmy_mesh->mesh_size.x2min;


      Real rotation_region_y_min = (L/2.0 - length_of_rotation_region/2.0) +  pmy_mesh->mesh_size.x2min;
      Real rotation_region_y_max = rotation_region_y_min + length_of_rotation_region;

      Real Bin = ( Bh * Bc * std::sin(theta_rot) ) / std::sqrt( SQR(Bh) + SQR(Bc) + 2.0*Bh*Bc*std::cos(theta_rot) ) ;
      Real Bhx = Bin;
      Real Bcx = - Bhx;


      Real sign_flip = 1.0;
      if (std::cos(theta_rot)<0.0) sign_flip=-1.0;
      Real Bhz = sign_flip * std::sqrt( SQR(Bh) - SQR(Bin) );
      Real Bcz =             std::sqrt( SQR(Bc) - SQR(Bin) );


      Real Bhx_norm = Bhx/Bh;
      Real Bcx_norm = Bcx/Bc;
      Real Bhz_norm = Bhz/Bh;
      Real Bcz_norm = Bcz/Bc;


      // I DON"T THINK THIS MAKES SENSE. Rotate angle, not linearly
      Real Bx_slope_norm = (Bcx_norm - Bhx_norm) / ( length_of_rotation_region) ; 
      Real Bz_slope_norm = (Bcz_norm - Bhz_norm) / ( length_of_rotation_region) ; 

      // Real angle_with_x_h = std::arctan2(Bhz,Bhx);
      // Real angle_with_x_c = std::arctan2(Bcz,Bzx);

      // Real Bx_slope = (Bcx - Bhx) / ( length_of_rotation_region) ; 
      // Real Bz_slope = (Bcz - Bhz) / ( length_of_rotation_region) ; 

      Real Bx, Bz;


      
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          pcoord->Face1Metric(k, j, il, iu+1, g, gi);
          for (int i=is; i<=ie+1; i++) {

            Real exp_arg_term,Bmag;
            if (pcoord->x2v(j) > 0.0){ // cold
              exp_arg_term = grav_acc / sigma_c * (2.0 + gamma/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              Bmag = Bc * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

            }
            else{ // hot
              exp_arg_term = grav_acc / sigma_h * (2.0 + gamma/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              Bmag = Bh * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));
            }


            if (pcoord->x2v(j) < rotation_region_y_min){
              Bx = Bhx * Bmag/Bh;
              Bz = Bhz * Bmag/Bh;
            }
            else if (pcoord->x2v(j) < L/2.0 + pmy_mesh->mesh_size.x2min){
              // Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              // Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              // Bx = Bx * Bh/B_norm;
              // Bz = Bz * Bh/B_norm;
              Bx = Bx * Bmag/B_norm;
              Bz = Bz * Bmag/B_norm;
              }
            else if (pcoord->x2v(j) < rotation_region_y_max){
              // Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              // Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);

              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              // Bx = Bx * Bc/B_norm;
              // Bz = Bz * Bc/B_norm;
              Bx = Bx * Bmag/B_norm;
              Bz = Bz * Bmag/B_norm;
            }
            else{
              Bx = Bcx * Bmag/Bc;
              Bz = Bcz * Bmag/Bc;
            }

            // if (pcoord->x2v(j) < (L/2.0  +  pmy_mesh->mesh_size.x2min) ){
            //   Bx = Bhx * Bmag/Bh;
            //   Bz = Bhz * Bmag/Bh;
            // }
            // else{
            //   Bx = Bcx * Bmag/Bc;
            //   Bz = Bcz * Bmag/Bc;
            // }

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
          pcoord->Face3Metric(k, j, il, iu, g, gi);
          for (int i=is; i<=ie; i++) {


            Real exp_arg_term,Bmag;
            if (pcoord->x2v(j) > 0.0){ // cold
              exp_arg_term = grav_acc / sigma_c * (2.0 + gamma/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              Bmag = Bc * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

            }
            else{ // hot
              exp_arg_term = grav_acc / sigma_h * (2.0 + gamma/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              Bmag = Bh * std::sqrt(( std::exp(pcoord->x2v(j)*exp_arg_term)));
            }

            if (pcoord->x2v(j) < rotation_region_y_min){
              Bx = Bhx * Bmag/Bh;
              Bz = Bhz * Bmag/Bh;
            }
            else if (pcoord->x2v(j) < L/2.0 + pmy_mesh->mesh_size.x2min){
              // Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              // Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              // Bx = Bx * Bh/B_norm;
              // Bz = Bz * Bh/B_norm;
              Bx = Bx * Bmag/B_norm;
              Bz = Bz * Bmag/B_norm;
              }
            else if (pcoord->x2v(j) < rotation_region_y_max){
              // Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_y_min);
              // Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_y_min);

              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);
              Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_y_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              // Bx = Bx * Bc/B_norm;
              // Bz = Bz * Bc/B_norm;
              Bx = Bx * Bmag/B_norm;
              Bz = Bz * Bmag/B_norm;
            }
            else{
              Bx = Bcx * Bmag/Bc;
              Bz = Bcz * Bmag/Bc;
            }



            // if (pcoord->x2v(j) < (L/2.0  +  pmy_mesh->mesh_size.x2min) ){
            //   Bx = Bhx * Bmag/Bh;
            //   Bz = Bhz * Bmag/Bh;
            // }
            // else{
            //   Bx = Bcx * Bmag/Bc;
            //   Bz = Bcz * Bmag/Bc;
            // }
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
    pin->GetReal("problem", "grav_acc");
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pcoord->CellMetric(k, j, il, iu, g, gi);
        for (int i=is; i<=ie; i++) {

          // Real L = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
          Real den=1.0;
          Real dh = 1.0;
          Real dc = dh * drat;
          if (pcoord->x3v(k) > 0.0) den *= drat;



          Real exp_arg_term,press,Bmag;
          if (pcoord->x3v(k) > 0.0){ // cold
            exp_arg_term = grav_acc / sigma_c * (2.0 + gamma/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
            press = press_over_rho_interface*dc * std::exp(pcoord->x3v(k)*exp_arg_term);
            den = dc * std::exp(pcoord->x3v(k)*exp_arg_term);
            Bmag = Bc * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));

          }
          else{ // hot
            exp_arg_term = grav_acc / sigma_h * (2.0 + gamma/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
            press = press_over_rho_interface*dh * std::exp(pcoord->x3v(k)*exp_arg_term);
            den = dh * std::exp(pcoord->x3v(k)*exp_arg_term);
            Bmag = Bh * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));
          }


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

          Real Lorentz = 1.0/std::sqrt(-g(I00) - SQR(v3));

          // Real Lorentz = 1.0/std::sqrt(-g(I00) - SQR(v2));
          Real uu0 = Lorentz;
          Real uu1 = 0.0;
          Real uu2 = 0.0;
          Real uu3 = v3 * Lorentz;

          Real uu1_ = uu1 - gi(I01,i) / gi(I00,i) * uu0;
          Real uu2_ = uu2 - gi(I02,i) / gi(I00,i) * uu0;
          Real uu3_ = uu3 - gi(I03,i) / gi(I00,i) * uu0;

          phydro->w(IM3,k,j,i) = uu3_;
          if (NON_BAROTROPIC_EOS) {
            phydro->w(IPR,k,j,i) =  press;

            // phydro->w(IPR,k,j,i) =  press_over_rho_interface*den + grav_acc*den*(pcoord->x3v(k));
          }
        }
      }
    }

    // initialize interface B
    if (MAGNETIC_FIELDS_ENABLED) {
      // Read magnetic field strength, angle [in degrees, 0 is along +ve X-axis]
      Real theta_rot = pin->GetReal("problem","theta_rot");
      theta_rot = (theta_rot/180.)*PI;

      Real rotation_region_z_min = (L/2.0-length_of_rotation_region/2.0) +  pmy_mesh->mesh_size.x3min;
      Real rotation_region_z_max = rotation_region_z_min + length_of_rotation_region;



      // Real rotation_region_z_min = 3.0*L/8.0 +  pmy_mesh->mesh_size.x3min;
      // Real rotation_region_z_max = rotation_region_z_min + L/4.0;

      Real Bin = ( Bh * Bc * std::sin(theta_rot) ) / std::sqrt( SQR(Bh) + SQR(Bc) + 2.0*Bh*Bc*std::cos(theta_rot) ) ;
      Real Bhx = Bin;
      Real Bcx = - Bhx;

      Real sign_flip = 1.0;
      if (std::cos(theta_rot)<0.0) sign_flip=-1.0;
      Real Bhy = sign_flip * std::sqrt( SQR(Bh) - SQR(Bin) );
      Real Bcy =             std::sqrt( SQR(Bc) - SQR(Bin) );

      // Real Bx_slope = (Bcx - Bhx) / ( length_of_rotation_region) ; 
      // Real By_slope = (Bcy - Bhy) / ( length_of_rotation_region) ; 



      Real Bhx_norm = Bhx/Bh;
      Real Bcx_norm = Bcx/Bc;
      Real Bhy_norm = Bhy/Bh;
      Real Bcy_norm = Bcy/Bc;


      // I DON"T THINK THIS MAKES SENSE. Rotate angle, not linearly
      Real Bx_slope_norm = (Bcx_norm - Bhx_norm) / ( length_of_rotation_region) ; 
      Real By_slope_norm = (Bcy_norm - Bhy_norm) / ( length_of_rotation_region) ;

      Real Bx, By;
      // angle = (angle/180.)*PI;

      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          pcoord->Face1Metric(k, j, il, iu+1, g, gi);
          for (int i=is; i<=ie+1; i++) {

            Real exp_arg_term,Bmag;
            if (pcoord->x3v(k) > 0.0){ // cold
              exp_arg_term = grav_acc / sigma_c * (2.0 + gamma/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              Bmag = Bc * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));

            }
            else{ // hot
              exp_arg_term = grav_acc / sigma_h * (2.0 + gamma/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              Bmag = Bh * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));
            }


            if (pcoord->x3v(k) < rotation_region_z_min){
              Bx = Bhx * Bmag/Bh;
              By = Bhy * Bmag/Bh;
            }
            else if (pcoord->x3v(k) < L/2.0 + pmy_mesh->mesh_size.x3min){
              // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);


              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bmag/B_norm;
              By = By * Bmag/B_norm;
              }
            else if (pcoord->x3v(k) < rotation_region_z_max){
              // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bmag/B_norm;
              By = By * Bmag/B_norm;
            }
            else{
              Bx = Bcx * Bmag/Bc;
              By = Bcy * Bmag/Bc;
            }
   
            pfield->b.x1f(k,j,i) = Bx;
          }
        }
      }
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
          pcoord->Face2Metric(k, j, il, iu+1, g, gi);
          for (int i=is; i<=ie; i++) {

            Real exp_arg_term,Bmag;
            if (pcoord->x3v(k) > 0.0){ // cold
              exp_arg_term = grav_acc / sigma_c * (2.0 + gamma/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              Bmag = Bc * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));

            }
            else{ // hot
              exp_arg_term = grav_acc / sigma_h * (2.0 + gamma/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              Bmag = Bh * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));
            }


            if (pcoord->x3v(k) < rotation_region_z_min){
              Bx = Bhx * Bmag/Bh;
              By = Bhy * Bmag/Bh;
            }
            else if (pcoord->x3v(k) < L/2.0 + pmy_mesh->mesh_size.x3min){
              // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bmag/B_norm;
              By = By * Bmag/B_norm;
              }
            else if (pcoord->x3v(k) < rotation_region_z_max){
              // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_z_min);
              // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_z_min);

              Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);
              By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_z_min);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bmag/B_norm;
              By = By * Bmag/B_norm;
            }
            else{
              Bx = Bcx * Bmag/Bc;
              By = Bcy * Bmag/Bc;
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






void linear_metric_3D(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  grav_acc = pin->GetReal("problem", "grav_acc");

  Real z0 = pin->GetReal("problem", "z0");

  Real Phi = grav_acc*(z-z0);

  // Set covariant components
  g(I00) = -(1.0 -2.0*Phi);
  g(I01) = 0;
  g(I02) = 0;
  g(I03) = 0;
  g(I11) = 1.0;
  g(I12) = 0;
  g(I13) = 0;
  g(I22) = 1.0 ;
  g(I23) = 0;
  g(I33) = 1.0 ;




  // // Set contravariant components
  g_inv(I00) = 1.0/g(I00);
  g_inv(I01) = 0;
  g_inv(I02) = 0;
  g_inv(I03) = 0;
  g_inv(I11) = 1.0;
  g_inv(I12) = 0;
  g_inv(I13) = 0;
  g_inv(I22) = 1.0;
  g_inv(I23) = 0.0;
  g_inv(I33) = 1.0;




  // Set x-derivatives of covariant components
  dg_dx1(I00) = 0;
  dg_dx1(I01) = 0;
  dg_dx1(I02) = 0;
  dg_dx1(I03) = 0;
  dg_dx1(I11) = 0;
  dg_dx1(I12) = 0;
  dg_dx1(I13) = 0;
  dg_dx1(I22) = 0;
  dg_dx1(I23) = 0;
  dg_dx1(I33) = 0;

  // Set y-derivatives of covariant components
  dg_dx2(I00) = 0;
  dg_dx2(I01) = 0;
  dg_dx2(I02) = 0;
  dg_dx2(I03) = 0;
  dg_dx2(I11) = 0;
  dg_dx2(I12) = 0;
  dg_dx2(I13) = 0;
  dg_dx2(I22) = 0;
  dg_dx2(I23) = 0;
  dg_dx2(I33) = 0;

  // Set phi-derivatives of covariant components
  dg_dx3(I00) = 2.0 * grav_acc;
  dg_dx3(I01) = 0;
  dg_dx3(I02) = 0;
  dg_dx3(I03) = 0;
  dg_dx3(I11) = 0;
  dg_dx3(I12) = 0;
  dg_dx3(I13) = 0;
  dg_dx3(I22) = 0;
  dg_dx3(I23) = 0;
  dg_dx3(I33) = 0;



  return;
}

void linear_metric_2D(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  grav_acc = pin->GetReal("problem", "grav_acc");


  Real y0 = pin->GetReal("problem", "y0");
  Real Phi = grav_acc*(y-y0);
  // Set covariant components
  g(I00) = -(1.0 -2.0*Phi);
  g(I01) = 0;
  g(I02) = 0;
  g(I03) = 0;
  g(I11) = 1.0;
  g(I12) = 0;
  g(I13) = 0;
  g(I22) = 1.0 ;
  g(I23) = 0;
  g(I33) = 1.0 ;




  // // Set contravariant components
  g_inv(I00) = 1.0/g(I00);
  g_inv(I01) = 0;
  g_inv(I02) = 0;
  g_inv(I03) = 0;
  g_inv(I11) = 1.0;
  g_inv(I12) = 0;
  g_inv(I13) = 0;
  g_inv(I22) = 1.0;
  g_inv(I23) = 0.0;
  g_inv(I33) = 1.0;




  // Set x-derivatives of covariant components
  dg_dx1(I00) = 0;
  dg_dx1(I01) = 0;
  dg_dx1(I02) = 0;
  dg_dx1(I03) = 0;
  dg_dx1(I11) = 0;
  dg_dx1(I12) = 0;
  dg_dx1(I13) = 0;
  dg_dx1(I22) = 0;
  dg_dx1(I23) = 0;
  dg_dx1(I33) = 0;

  // Set y-derivatives of covariant components
  dg_dx2(I00) = 2.0 * grav_acc;
  dg_dx2(I01) = 0;
  dg_dx2(I02) = 0;
  dg_dx2(I03) = 0;
  dg_dx2(I11) = 0;
  dg_dx2(I12) = 0;
  dg_dx2(I13) = 0;
  dg_dx2(I22) = 0;
  dg_dx2(I23) = 0;
  dg_dx2(I33) = 0;

  // Set phi-derivatives of covariant components
  dg_dx3(I00) = 0;
  dg_dx3(I01) = 0;
  dg_dx3(I02) = 0;
  dg_dx3(I03) = 0;
  dg_dx3(I11) = 0;
  dg_dx3(I12) = 0;
  dg_dx3(I13) = 0;
  dg_dx3(I22) = 0;
  dg_dx3(I23) = 0;
  dg_dx3(I33) = 0;



  return;
}
