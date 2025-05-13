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

Real DivergenceB(MeshBlock *pmb, int iout);
Real vsq(MeshBlock *pmb, int iout);
// Real cs;

void linear_metric_2D(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
void linear_metric_3D(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);

Real Phi_func(Real z, Real gravitational_acceleration,Real z0 );
void rungeKutta4(
    void (*f)(Real t, Real y, bool is_top, ParameterInput *pin, MeshBlock *pmb,Real *dydt),
    Real *y,
    Real t0,
    Real t1,
    Real dt,
    bool is_top,
    ParameterInput *pin,
    MeshBlock *pmb
);


namespace {
// made global to share with BC functions
Real grav_acc,shear_velocity;
Real beta_c, sigma_c,press_over_rho_interface, sigma_h,beta_h;
Real rho_h,rho_c, Bh,Bc,drat;
Real P_h, P_c;
Real L, length_of_rotation_region;
Real rotation_region_min, rotation_region_max;
Real theta_rot;
} // namespace

int RefinementCondition(MeshBlock *pmb);


Real GetBAngle(const Real x){

      Real Bin = ( Bh * Bc * std::sin(theta_rot) ) / std::sqrt( SQR(Bh) + SQR(Bc) + 2.0*Bh*Bc*std::cos(theta_rot) ) ;
      Real Bhx = Bin;
      Real Bcx = - Bhx;


      Real sign_flip = 1.0;
      if (std::cos(theta_rot)<0.0) sign_flip=-1.0;
      Real Bhz = sign_flip * std::sqrt( SQR(Bh) - SQR(Bin) );
      Real Bcz =             std::sqrt( SQR(Bc) - SQR(Bin) );


      Real angle_with_x_h = std::atan2(Bhz,Bhx);
      Real angle_with_x_c = std::atan2(Bcz,Bcx);


      Real w = (x - rotation_region_min) / (rotation_region_max - rotation_region_min);
      if (w>1) w = 1.0;
      if (w<0) w = 0.0;
      // Real theta_y = (1.0 - w) * angle_with_x_h + w * angle_with_x_c;

      Real delta_theta = angle_with_x_c - angle_with_x_h;
      // Wrap to [-π, π]
      if (delta_theta > PI) delta_theta -= 2.0 * PI;
      if (delta_theta < -PI) delta_theta += 2.0 * PI;


      return  angle_with_x_h + w * delta_theta;
}

void Pressure_ODE_2D(Real t, Real y, bool is_top,ParameterInput *pin, MeshBlock *pmb, Real *dydt) {

      Real P = y;
      Real y0 = pin->GetReal("problem", "y0");
      Real Phi_N = -grav_acc * (t-y0);
      Real gamma_adi = pmb->peos->GetGamma();



      Real g_00 = -(1.0 + 2.0*Phi_N);
      Real g_11 = 1.0;
      Real g_22 = 1.0 - 2.0*Phi_N;
      Real g_02 = -2.0*Phi_N ;
      Real g_20 = g_02;
      Real g_33 = 1.0;

      Real v_x;
      v_x = shear_velocity/2.0;
      Real v_y = 0.0;
      Real v_z = 0.0;


      Real uu_t = std::sqrt( -1 / ( g_00 + g_11*SQR(v_x) + g_22*SQR(v_y) + g_33*SQR(v_z) + 
                                2.0*g_02*v_y )   );
      Real uu_x = uu_t*v_x;
      Real uu_y = uu_t*v_y;
      Real uu_z = uu_t*v_z;

      Real theta_y = GetBAngle(t);

      Real Bx_over_fake_B = std::cos(theta_y);
      Real By_over_fake_B = 0.0;
      Real Bz_over_fake_B = std::sin(theta_y);


      // By_over_Bx = 0
      // Bz_over_Bx = 0

      Real b0_over_fake_B = g_11 * uu_x * Bx_over_fake_B + g_22 * By_over_fake_B * uu_y  + g_02 * By_over_fake_B * uu_t + g_33 * Bz_over_fake_B * uu_z ;
      Real bu_over_fake_B[4];

      bu_over_fake_B[0] = b0_over_fake_B;
      bu_over_fake_B[1] = (Bx_over_fake_B + b0_over_fake_B * uu_x)/uu_t;
      bu_over_fake_B[2] = (By_over_fake_B + b0_over_fake_B * uu_y)/uu_t;
      bu_over_fake_B[3] = (Bz_over_fake_B + b0_over_fake_B * uu_z)/uu_t;

      Real bsq_over_fake_B_sq = bu_over_fake_B[0]*bu_over_fake_B[0]*g_00 + bu_over_fake_B[1]*bu_over_fake_B[1]*g_11 + bu_over_fake_B[2]*bu_over_fake_B[2] *g_22  + bu_over_fake_B[0]*bu_over_fake_B[2] *g_02 + bu_over_fake_B[2]*bu_over_fake_B[0] *g_20 + bu_over_fake_B[3]*bu_over_fake_B[3]*g_33 ;

      Real numerator = grav_acc;
      Real denominator = (1.0 + 2.0 * Phi_N - SQR(v_x) );
      Real prefactor = numerator / denominator;

      Real beta, sigma;

      if (is_top){
        beta = beta_c;
        sigma = sigma_c;
      }
      else{
        beta = beta_h;
        sigma = sigma_h;
      }
      Real bracket = (
          2 / sigma +
          (gamma_adi * beta) / (gamma_adi - 1) +
          2  - 
          2 * (SQR(Bx_over_fake_B)/bsq_over_fake_B_sq) * ( SQR(v_x) )
      );
      *dydt =  (P / (beta + 1)) * prefactor * bracket;


}
void Pressure_ODE_3D(Real t, Real y, bool is_top,ParameterInput *pin, MeshBlock *pmb, Real *dydt) {


      Real P = y;
      Real z0 = pin->GetReal("problem", "z0");
      Real Phi_N = -grav_acc * (t-z0);
      Real gamma_adi = pmb->peos->GetGamma();



      Real g_00 = -(1.0 + 2.0*Phi_N);
      Real g_11 = 1.0;
      Real g_33 = 1.0 - 2.0*Phi_N;
      Real g_03 = -2.0*Phi_N ;
      Real g_30 = g_03;
      Real g_22 = 1.0;

      Real v_x;
      v_x = shear_velocity/2.0;
      Real v_y = 0.0;
      Real v_z = 0.0;


      Real uu_t = std::sqrt( -1 / ( g_00 + g_11*SQR(v_x) + g_22*SQR(v_y) + g_33*SQR(v_z) + 
                                2.0*g_03*v_z )   );
      Real uu_x = uu_t*v_x;
      Real uu_y = uu_t*v_y;
      Real uu_z = uu_t*v_z;

      Real theta_y = GetBAngle(t);

      Real Bx_over_fake_B = std::cos(theta_y);
      Real By_over_fake_B = std::sin(theta_y);
      Real Bz_over_fake_B = 0.0;


      // By_over_Bx = 0
      // Bz_over_Bx = 0

      Real b0_over_fake_B = g_11 * uu_x * Bx_over_fake_B + g_22 * By_over_fake_B * uu_y  + g_03 * Bz_over_fake_B * uu_t + g_33 * Bz_over_fake_B * uu_z;
      Real bu_over_fake_B[4];

      bu_over_fake_B[0] = b0_over_fake_B;
      bu_over_fake_B[1] = (Bx_over_fake_B + b0_over_fake_B * uu_x)/uu_t;
      bu_over_fake_B[2] = (By_over_fake_B + b0_over_fake_B * uu_y)/uu_t;
      bu_over_fake_B[3] = (Bz_over_fake_B + b0_over_fake_B * uu_z)/uu_t;

      Real bsq_over_fake_B_sq = bu_over_fake_B[0]*bu_over_fake_B[0]*g_00 + bu_over_fake_B[1]*bu_over_fake_B[1]*g_11 + bu_over_fake_B[2]*bu_over_fake_B[2] *g_22  + bu_over_fake_B[0]*bu_over_fake_B[3] *g_03 + bu_over_fake_B[3]*bu_over_fake_B[0] *g_30 + bu_over_fake_B[3]*bu_over_fake_B[3]*g_33 ;

      Real numerator = grav_acc;
      Real denominator = (1.0 + 2.0 * Phi_N - SQR(v_x) );
      Real prefactor = numerator / denominator;

      Real beta, sigma;

      if (is_top){
        beta = beta_h;
        sigma = sigma_h;
      }
      else{
        beta = beta_c;
        sigma = sigma_c;
      }
      Real bracket = (
          2 / sigma +
          (gamma_adi * beta) / (gamma_adi - 1) +
          2  - 
          2 * (SQR(Bx_over_fake_B)/bsq_over_fake_B_sq) * ( SQR(v_x) )
      );
      *dydt =  (P / (beta + 1)) * prefactor * bracket;
}

// Runge-Kutta 4th order ODE solver 
void rungeKutta4(
    void (*f)(Real t, Real y, bool is_top, ParameterInput *pin, MeshBlock *pmb,Real *dydt),
    Real *y,
    Real t0,
    Real t1,
    Real dt,
    bool is_top,
    ParameterInput *pin,
    MeshBlock *pmb
) {
    Real t = t0;
    Real  k1, k2, k3, k4, yTemp;

    Real dt_temp = dt;

    if (t0<t1){

      // int n_loop = 0;

    while (t < t1) {

        // fprintf(stderr,"n_loop: %d t0: %g t1: %g t: %g dt: %g \n",n_loop, t0,t1,t,dt);
        // n_loop += 1;

        f(t, *y, is_top, pin, pmb, &k1);

        yTemp = *y + dt * k1 / 2.0;
        f(t + dt_temp / 2.0, yTemp, is_top, pin, pmb, &k2);

        yTemp = *y + dt * k2 / 2.0;
        f(t + dt_temp / 2.0, yTemp, is_top, pin, pmb, &k3);

        yTemp = *y + dt * k3;
        f(t + dt_temp, yTemp, is_top, pin, pmb, &k4);

        // Update y
        *y += dt_temp / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        t += dt_temp;

        if (t + dt_temp > t1) dt_temp = t1-t;

      }

    }

    else {

      // int n_loop = 0;
    while (t > t1) {

        // fprintf(stderr,"n_loop: %d t0: %g t1: %g t: %g dt: %g \n",n_loop, t0,t1,t,dt);
        // n_loop += 1;
        f(t, *y, is_top, pin, pmb, &k1);

        yTemp = *y + dt * k1 / 2.0;
        f(t + dt_temp / 2.0, yTemp, is_top, pin, pmb, &k2);

        yTemp = *y + dt * k2 / 2.0;
        f(t + dt_temp / 2.0, yTemp, is_top, pin, pmb, &k3);

        yTemp = *y + dt * k3;
        f(t + dt_temp, yTemp, is_top, pin, pmb, &k4);

        // Update y
        *y += dt_temp / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        t += dt_temp;

        if (t + dt_temp < t1) dt_temp = t1-t;

      }

    }
}

void integrate_P_ODE(int il, int iu, int jl, int ju, int kl, int ku, AthenaArray<Real> x_coord,MeshBlock *pmb,ParameterInput *pin,AthenaArray<Real> &P_sol ){



 //Integrate the pressure eqation by starting at the interface and then integrated by roughly 10 times 
 //smaller steps

  if (pmb->block_size.nx3 > 1) {  //3D

    if (x_coord(kl) > 0.0){ //whole block is above y=0
       //do first step
       Real dt_runge_kutta = (x_coord(kl)-0.0)/(pmb->pmy_mesh->mesh_size.nx3*10.0);
       Real P_result = P_c;
       rungeKutta4(Pressure_ODE_3D,&P_result, 0.0,x_coord(kl), dt_runge_kutta, true, pin,pmb); 
       P_sol(kl) = P_result;

       for (int k=kl+1; k<=ku; k++) {
         dt_runge_kutta = (x_coord(k)-x_coord(k-1))/(10.0);
         P_result = P_sol(k-1);
         rungeKutta4(Pressure_ODE_3D, &P_result, x_coord(k-1),x_coord(k), dt_runge_kutta, true, pin,pmb); 
         P_sol(k) = P_result;

       }

    }

    else if (x_coord(ku) < 0.0){ //whole block is below y=0

      //do first step
       Real dt_runge_kutta = (x_coord(ku) - 0.0)/(pmb->pmy_mesh->mesh_size.nx3*10.0);
       Real P_result = P_h;
       rungeKutta4(Pressure_ODE_3D,&P_result, 0.0, x_coord(ku),  dt_runge_kutta, false, pin,pmb); 
       P_sol(ku) = P_result;


        for (int k=ku-1; k>=kl; k--) {
         dt_runge_kutta = (x_coord(k)-x_coord(k+1))/(10.0);
         P_result = P_sol(k+1);
         rungeKutta4(Pressure_ODE_3D, &P_result, x_coord(k+1),x_coord(k), dt_runge_kutta, false, pin,pmb); 
         P_sol(k) = P_result;

       }

    }

    else{ //mixed case

      //first find index where transition happens
      int k_trans = kl;
      for (int k=kl; k<=ku; k++) {
        if (x_coord(k) >0.0){
          k_trans = k;
          break;
        }
      }

      //do upper first

       Real dt_runge_kutta = (x_coord(k_trans)-0.0)/(pmb->pmy_mesh->mesh_size.nx3*10.0);
       Real P_result = P_c;
       rungeKutta4(Pressure_ODE_3D,&P_result, 0.0,x_coord(k_trans), dt_runge_kutta, true, pin,pmb); 
       P_sol(k_trans) = P_result;

       for (int k=k_trans+1; k<=ku; k++) {
         dt_runge_kutta = (x_coord(k)-x_coord(k-1))/(10.0);
         P_result = P_sol(k-1);
         rungeKutta4(Pressure_ODE_3D, &P_result, x_coord(k-1),x_coord(k), dt_runge_kutta, true, pin,pmb); 
         P_sol(k) = P_result;

       }

       // now lower
       dt_runge_kutta = (x_coord(k_trans-1) - 0.0)/(pmb->pmy_mesh->mesh_size.nx3*10.0);
       P_result = P_h;
       rungeKutta4(Pressure_ODE_3D,&P_result, 0.0, x_coord(k_trans-1),  dt_runge_kutta, false,pin,pmb); 
       P_sol(k_trans-1) = P_result;


        for (int k=k_trans-2; k>=kl; k--) {
         dt_runge_kutta = (x_coord(k)-x_coord(k+1))/(10.0);
         P_result = P_sol(k+1);
         rungeKutta4(Pressure_ODE_3D, &P_result, x_coord(k+1),x_coord(k), dt_runge_kutta, false, pin,pmb); 
         P_sol(k) = P_result;

       }



    }
  }
  else{ //2D

    if (x_coord(jl) > 0.0){ //whole block is above y=0
       //do first step
       Real dt_runge_kutta = (x_coord(jl)-0.0)/(pmb->pmy_mesh->mesh_size.nx2*10.0);
       Real P_result = P_c;
       rungeKutta4(Pressure_ODE_2D,&P_result, 0.0,x_coord(jl), dt_runge_kutta, true, pin,pmb); 
       P_sol(jl) = P_result;

       for (int j=jl+1; j<=ju; j++) {
         dt_runge_kutta = (x_coord(j)-x_coord(j-1))/(10.0);
         P_result = P_sol(j-1);
         rungeKutta4(Pressure_ODE_2D, &P_result, x_coord(j-1),x_coord(j), dt_runge_kutta, true, pin,pmb); 
         P_sol(j) = P_result;

       }

    }

    else if (x_coord(ju) < 0.0){ //whole block is below y=0

      //do first step
       Real dt_runge_kutta = (x_coord(ju) - 0.0)/(pmb->pmy_mesh->mesh_size.nx2*10.0);
       Real P_result = P_h;
       rungeKutta4(Pressure_ODE_2D,&P_result, 0.0, x_coord(ju),  dt_runge_kutta, false, pin,pmb); 
       P_sol(ju) = P_result;


        for (int j=ju-1; j>=jl; j--) {
         dt_runge_kutta = (x_coord(j)-x_coord(j+1))/(10.0);
         P_result = P_sol(j+1);
         rungeKutta4(Pressure_ODE_2D, &P_result, x_coord(j+1),x_coord(j), dt_runge_kutta, false, pin,pmb); 
         P_sol(j) = P_result;

       }

    }

    else{ //mixed case

      //first find index where transition happens
      int j_trans = jl;
      for (int j=jl; j<=ju; j++) {
        if (x_coord(j) >0.0){
          j_trans = j;
          break;
        }
      }

      //do upper first

       Real dt_runge_kutta = (x_coord(j_trans)-0.0)/(pmb->pmy_mesh->mesh_size.nx2*10.0);
       Real P_result = P_c;
       rungeKutta4(Pressure_ODE_2D,&P_result, 0.0,x_coord(j_trans), dt_runge_kutta, true, pin,pmb); 
       P_sol(j_trans) = P_result;

       for (int j=j_trans+1; j<=ju; j++) {
         dt_runge_kutta = (x_coord(j)-x_coord(j-1))/(10.0);
         P_result = P_sol(j-1);
         rungeKutta4(Pressure_ODE_2D, &P_result, x_coord(j-1),x_coord(j), dt_runge_kutta, true, pin,pmb); 
         P_sol(j) = P_result;

       }

       // now lower
       dt_runge_kutta = (x_coord(j_trans-1) - 0.0)/(pmb->pmy_mesh->mesh_size.nx2*10.0);
       P_result = P_h;
       rungeKutta4(Pressure_ODE_2D,&P_result, 0.0, x_coord(j_trans-1),  dt_runge_kutta, false,pin,pmb); 
       P_sol(j_trans-1) = P_result;


        for (int j=j_trans-2; j>=jl; j--) {
         dt_runge_kutta = (x_coord(j)-x_coord(j+1))/(10.0);
         P_result = P_sol(j+1);
         rungeKutta4(Pressure_ODE_2D, &P_result, x_coord(j+1),x_coord(j), dt_runge_kutta, false, pin,pmb); 
         P_sol(j) = P_result;

       }



  }

}

}

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



    grav_acc = pin->GetReal("problem", "grav_acc");

    theta_rot = pin->GetReal("problem","theta_rot");
    theta_rot = (theta_rot/180.)*PI;

    Real f_kep_shear = pin->GetOrAddReal("problem","f_kep_shear",0.0);
    Real effective_graviational_radius = std::sqrt(1.0/std::fabs(grav_acc));
    Real effective_keplerian_velocity = std::sqrt(1/effective_graviational_radius);
    shear_velocity = f_kep_shear * effective_keplerian_velocity;


    beta_c = pin->GetOrAddReal("problem","beta_c",1.0);
    sigma_c = pin->GetOrAddReal("problem","sigma_c",1.0);


    press_over_rho_interface = beta_c * sigma_c /2.0;
    sigma_h = pin->GetOrAddReal("problem","sigma_h",1.0);
    beta_h = press_over_rho_interface/sigma_h * 2.0;


    rho_h = 1.0;


    Bh = std::sqrt(sigma_h * rho_h);

    // sigma_h/sigma_c = Bh^2/Bc^2 * drat
    // Bh^2/Bc^2 = 1 + (1 - 1/drat)*beta_c 
    // sigma_h/sigma_c  = drat + (drat -1)*beta_c
    drat = ( sigma_h/sigma_c+beta_c )/(1.0 + beta_c);

    rho_c = rho_h * drat;


    P_h = press_over_rho_interface * rho_h;
    P_c = press_over_rho_interface * rho_c;

    Bc = Bh / std::sqrt(1.0 + (1.0 - 1.0/drat)*beta_c);

    if (mesh_size.nx3==1) L = mesh_size.x2max - mesh_size.x2min;
    else L = mesh_size.x3max - mesh_size.x3min;
    length_of_rotation_region = pin->GetOrAddReal("problem","length_of_rotation_region",L/10.0);

    if (mesh_size.nx3==1) {
      rotation_region_min = (L/2.0 - length_of_rotation_region/2.0) +  mesh_size.x2min;
      rotation_region_max = rotation_region_min + length_of_rotation_region;
    }
    else{
      rotation_region_min = (L/2.0 - length_of_rotation_region/2.0) +  mesh_size.x3min;
      rotation_region_max = rotation_region_min + length_of_rotation_region;
    }



    AllocateUserHistoryOutput(2);
    EnrollUserHistoryOutput(0,vsq,"vsq");  
    EnrollUserHistoryOutput(1,DivergenceB,"divb");


    if (mesh_size.nx3>1) EnrollUserMetric(linear_metric_3D);
    else EnrollUserMetric(linear_metric_2D);



  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate space for scratch arrays
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie + NGHOST + 1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie + NGHOST + 1);


  AllocateUserOutputVariables(3);

}



// v^2
Real vsq(MeshBlock *pmb, int iout)
{
  Real vsq=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;


  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0];
  AthenaArray<Real> &gi = pmb->ruser_meshblock_data[1];

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->CellMetric(k, j, is, ie, g, gi);
      for(int i=is; i<=ie; i++) {

           // Calculate normal-frame Lorentz factor
          Real uu1 = pmb->phydro->w(IVX,k,j,i);
          Real uu2 = pmb->phydro->w(IVY,k,j,i);
          Real uu3 = pmb->phydro->w(IVZ,k,j,i);
          Real tmp = g(I11,i) * SQR(uu1) + 2.0 * g(I12,i) * uu1 * uu2
              + 2.0 * g(I13,i) * uu1 * uu3 + g(I22,i) * SQR(uu2)
              + 2.0 * g(I23,i) * uu2 * uu3 + g(I33,i) * SQR(uu3);
          Real gamma = std::sqrt(1.0 + tmp);

          // Calculate 4-velocity
          Real alpha = std::sqrt(-1.0 / gi(I00,i));
          Real u0 = gamma / alpha;
          Real u1 = uu1 - alpha * gamma * gi(I01,i);
          Real u2 = uu2 - alpha * gamma * gi(I02,i);
          Real u3 = uu3 - alpha * gamma * gi(I03,i);

          Real v1 = u1/u0;
          Real v2 = u2/u0;
          Real v3 = u3/u0;


        Real v_shear = 0;
        if (pmb->block_size.nx3==1) {
          if (pmb->pcoord->x2v(j)>0) v_shear = shear_velocity;
        }
        else{
          if (pmb->pcoord->x3v(k)>0) v_shear = shear_velocity;
        }
        // vsq+= SQR( v1 -v_shear) + SQR( v2 ) + SQR( v3 );

        vsq+= SQR(v2) + SQR(v3);
      }
    }
  }

  return vsq;
}

Real Phi_func(Real z, Real gravitational_acceleration,Real z0 ){

  return -gravitational_acceleration * (z-z0);
}


/* Store some useful variables like mdot and vr */

Real DivergenceB(MeshBlock *pmb, int iout)
{
  Real divb=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pmb->pfield->b;

  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pmb->pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pmb->pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pmb->pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pmb->pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {
        divb+=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
      }
    }
  }

  face1.DeleteAthenaArray();
  face2p.DeleteAthenaArray();
  face2m.DeleteAthenaArray();
  face3p.DeleteAthenaArray();
  face3m.DeleteAthenaArray();

  return divb;
}


void MeshBlock::UserWorkInLoop() {


  // AthenaArray<Real> &g = ruser_meshblock_data[0];
  // AthenaArray<Real> &gi = ruser_meshblock_data[1];


  // int k = ks;
  // int j = js;
  // int i = is + 4;

  // pcoord->CellMetric(k, j, is, ie, g, gi);
  // Real g00 = g(I00,i);
  // Real g22 = g(I22,i);
  // pcoord->CellMetric(k, j-1, is, ie, g, gi);

  // Real g00p1 = g(I00,i);
  // Real g22p1 = g(I22,i);

  // pcoord->CellMetric(k, j-2, is, ie, g, gi);

  // Real g00p2 = g(I00,i);
  // Real g22p2 = g(I22,i);

  // if (pcoord->x2v(j)>0) fprintf(stderr,"x y z: %g %g %g \n i j k %d %d %d \n rho: %g %g %g \n v2: %g %g %g \n Bcc2: %g %g %g \n g00: %g %g %g \n g22: %g %g %g \n",
  //   pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k), i,j,k,
  //   phydro->w(IDN,k,j,i),phydro->w(IDN,k,j-1,i),phydro->w(IDN,k,j-2,i),
  //   phydro->w(IVY,k,j,i),phydro->w(IVY,k,j-1,i),phydro->w(IVY,k,j-2,i),
  //   pfield->bcc(IB2,k,j,i),pfield->bcc(IB2,k,j-1,i),pfield->bcc(IB2,k,j-2,i),
  //   g00,g00p1,g00p2,
  //   g22,g22p1,g22p2 );


  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pfield->b;

  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {
        Real divb=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));

        user_out_var(0,k,j,i) = divb;
        user_out_var(1,k,j,i) = b.x3f(k,j,i);
        user_out_var(2,k,j,i) = b.x3f(k+1,j,i);
      }
    }
  }

  face1.DeleteAthenaArray();
  face2p.DeleteAthenaArray();
  face2m.DeleteAthenaArray();
  face3p.DeleteAthenaArray();
  face3m.DeleteAthenaArray();
  return;
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
  Real gamma_adi = peos->GetGamma();
  Real gm1 = gamma_adi - 1.0;
  // Real press_over_rho = SQR(cs)/(gamma - (gamma/(gamma-1))*SQR(cs));
  Real kx = 2.0*(PI)/(pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min);
  Real ky = 2.0*(PI)/(pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min);
  Real kz = 2.0*(PI)/(pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min);
    
  
  // Read perturbation amplitude, problem switch, density ratio
  Real amp = pin->GetReal("problem","amp");
  int iprob = pin->GetInteger("problem","iprob");

  Real cs = std::sqrt(press_over_rho_interface * gamma_adi / (1.0 + gamma_adi/(gm1) *press_over_rho_interface) );

  // 2D PROBLEM ---------------------------------------------------------------

  if (block_size.nx3 == 1) {
    grav_acc = pin->GetReal("problem", "grav_acc");
    Real y0 = pin->GetReal("problem", "y0");

    AthenaArray<Real> P_sol;
    P_sol.NewAthenaArray(ju-jl+1);
    integrate_P_ODE(il,iu,jl,ju,kl,ku,pcoord->x2v,this,pin,P_sol );
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        pcoord->CellMetric(k, j, il, iu, g, gi);
        for (int i=il; i<=iu; i++) {
          Real dh = 1.0;
          Real dc = dh * drat;
          Real den=1.0;
          if (pcoord->x2v(j) > 0.0) den *= drat;

          // Real Phi_const = Phi/

          // P = P0 * ( B + Cy )^{A/C}

          //B + Cy = 1 + 2Phi
          // B + Cy = 1 - 2 grav_acc * (y-y0) = 1 +2 grav_acc*y0 - 2 grav_acc * y
          // B = 1 + 2 grav_acc*y0
          // C = - 2 grav_acc
          //A = exp_arg

          //P0 * (B^(A/C) = press_over_rho_interface*d
          //P0  = press_over_rho_interface*d

          Real v2 = 0;
          Real v1 = 0.0;
          v1 = shear_velocity/2.0;
          Real v3 = 0;


          // Real B_const = 1.0 + 2.0 * grav_acc*y0 - SQR(v1);
          // Real C_const = -2.0*grav_acc;

          // Real exp_arg_term,press,Bmag;
          // if (pcoord->x2v(j) > 0.0){ // cold
          //   exp_arg_term = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
          //   Real A_const = exp_arg_term;
          //   press = press_over_rho_interface*dc * std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const);
          //   // press = press_over_rho_interface*dc * std::exp(pcoord->x2v(j)*exp_arg_term);
          //   // den = dc * std::exp(pcoord->x2v(j)*exp_arg_term);
          //   // Bmag = Bc * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

          //   den = dc * std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const);
          //   Bmag = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const));
          // }
          // else{ // hot
          //   exp_arg_term = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
          //   Real A_const = exp_arg_term;
          //   press = press_over_rho_interface*dh * std::pow( 1.0 + C_const/B_const *pcoord->x2v(j), A_const/C_const);


          //   // press = press_over_rho_interface*dh * std::exp(pcoord->x2v(j)*exp_arg_term);
          //   // den = dh * std::exp(pcoord->x2v(j)*exp_arg_term);
          //   // Bmag = Bh * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));


          //   den = dh * std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const);
          //   Bmag = Bh * std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const);
          // }

          den = P_sol(j)/press_over_rho_interface;

          phydro->w(IDN,k,j,i) = den;


          // u^\mu u^\nu g_\mu \nu = -1 
          // u^\mu = dt/dTau (1, dx/dt, dy/dt, dz/dt) = u^0 (1, v1,v2,v3)
          // assume g_mu nu is diagonal
          // u^0^2 (g_00 + g_11 v1 + g_22 v2 + g_33 v3) = -1
          // u^0 = sqrt[ -1/( g_00 + g_11 v + g_22 v2 + g_33 v3) ]

          Real u0 = std::sqrt( -1 / ( g(I00,i) + g(I11,i)*SQR(v1) + g(I22,i)*SQR(v2) + g(I33,i)*SQR(v3) + 
                                      2.0*g(I01,i)*v1 + 2.0*g(I02,i)*v2 + 2.0*g(I03,i)*v3  )   ); 
          Real u1 = u0*v1;
          Real u2 = u0*v2;
          Real u3 = u0*v3;

          // Real Lorentz = 1.0/std::sqrt(-g(I00,i) - SQR(v2));
          // Real u0 = Lorentz;
          // Real u1 = 0.0;
          // Real u2 = v2*Lorentz;
          // Real u3 = 0.0;

          // Now convert to Athena++ velocities (see White+ 2016)
          Real uu1 = u1 - gi(I01,i) / gi(I00,i) * u0;
          Real uu2 = u2 - gi(I02,i) / gi(I00,i) * u0;
          Real uu3 = u3 - gi(I03,i) / gi(I00,i) * u0;

          phydro->w(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = uu3;
          if (NON_BAROTROPIC_EOS) {
            phydro->w(IEN,k,j,i) = P_sol(j);
            // phydro->w(IEN,k,j,i) = (press_over_rho_interface*den + grav_acc*den*(pcoord->x2v(j)));
            
          }
        }
      }
    }



    // initialize interface B, same for all iprob
    if (MAGNETIC_FIELDS_ENABLED) {
      // Read magnetic field strength, angle [in degrees, 0 is along +ve X-axis]
      // Real b0 = pin->GetReal("problem","b0");

      // Real L = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
      // Real rotation_region_min = 3.0*L/8.0 +  pmy_mesh->mesh_size.x2min;
      // Real rotation_region_max = rotation_region_min + L/4.0;

      // Real rotation_region_min =  L/2.0 + pmy_mesh->mesh_size.x2min;
      // Real rotation_region_max = L/2.0 + pmy_mesh->mesh_size.x2min;


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
      // Real Bx_slope_norm = (Bcx_norm - Bhx_norm) / ( length_of_rotation_region) ; 
      // Real Bz_slope_norm = (Bcz_norm - Bhz_norm) / ( length_of_rotation_region) ; 

      // Real angle_with_x_h = std::atan2(Bhz,Bhx);
      // Real angle_with_x_c = std::atan2(Bcz,Bzx);

      //Bhx = Bh*std::cos(angle_with_x_h)
      //Bhz = Bh*std::sin(angle_with_x_c)

      // Real Bx_slope = (Bcx - Bhx) / ( length_of_rotation_region) ; 
      // Real Bz_slope = (Bcz - Bhz) / ( length_of_rotation_region) ; 

      Real Bx, Bz,By;



      // Real C_const = -2.0*grav_acc;


      // Real exp_arg_term_rotation_region_ymax = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
      // Real A_const_rotation_region_ymax = exp_arg_term_rotation_region_ymax;
      // Real B_const_rotation_region_ymax = 1.0 + 2.0 * grav_acc*y0 - SQR(shear_velocity);

      // Real Bmag_rotation_region_ymax = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const_rotation_region_ymax *rotation_region_max, A_const_rotation_region_ymax/C_const));

      // Real Bx_rotation_region_ymax = Bcx * Bmag_rotation_region_ymax/Bc;
      // Real Bz_rotation_region_ymax = Bcz * Bmag_rotation_region_ymax/Bc;


      // Real exp_arg_term_rotation_region_ymin = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
      // Real A_const_rotation_region_ymin = exp_arg_term_rotation_region_ymin;
      // Real B_const_rotation_region_ymin = 1.0 + 2.0 * grav_acc*y0;


      // Real Bmag_rotation_region_ymin = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const_rotation_region_ymin *rotation_region_min, A_const_rotation_region_ymin/C_const));

      // Real Bx_rotation_region_ymin = Bhx * Bmag_rotation_region_ymin/Bh;
      // Real Bz_rotation_region_ymin = Bhz * Bmag_rotation_region_ymin/Bh;


      // Real Bx_slope = (Bx_rotation_region_ymax - Bx_rotation_region_ymin) / ( length_of_rotation_region) ; 
      // Real Bz_slope = (Bz_rotation_region_ymax - Bz_rotation_region_ymin) / ( length_of_rotation_region) ; 

      
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          pcoord->Face1Metric(k, j, il, iu+1, g, gi);
          for (int i=il; i<=iu+1; i++) {

            Real exp_arg_term,Bmag;

            if (pcoord->x2v(j) > 0.0){ // cold
              // exp_arg_term = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*y0 - SQR(shear_velocity);
              // press = press_over_rho_interface*dc * std::pow( 1.0+ C_const/B_const *y, A_const/C_const);

              // Bmag = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const));
              // Bmag = Bc * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

              Bmag =  std::sqrt(P_sol(j)/beta_c*2.0);

            }
            else{ // hot
              // exp_arg_term = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*y0;
              // // press = press_over_rho_interface*dc * std::pow( 1.0+ C_const/B_const *y, A_const/C_const);

              // Bmag = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const));

              // Bmag = Bh * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

              Bmag =  std::sqrt(P_sol(j)/beta_h*2.0);
            }


            if (pcoord->x2v(j) < rotation_region_min){
              Bx = Bhx * Bmag/Bh;
              Bz = Bhz * Bmag/Bh;
            }
            else if (pcoord->x2v(j) < rotation_region_max){
              // Real w = ( pcoord->x2v(j) -rotation_region_min )/(rotation_region_max-rotation_region_min);
              // // Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_min);
              // // Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_min);
              // // Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_min);
              // // Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_min);



              // // Bx = Bhx_norm * std::sin((1.0-w)*theta_rot)/(sin(theta_rot) + 1e-10) + Bcx_norm*std::sin(w*theta_rot)/(sin(theta_rot) + 1e-10);
              // // Bz = Bhz_norm * std::sin((1.0-w)*theta_rot)/(sin(theta_rot) + 1e-10) + Bcz_norm*std::sin(w*theta_rot)/(sin(theta_rot) + 1e-10);


              // Bx = Bx_rotation_region_ymin + Bx_slope * ( pcoord->x2v(j) - rotation_region_min);
              // Bz = Bz_rotation_region_ymin + Bz_slope * ( pcoord->x2v(j) - rotation_region_min);

              Real theta_b = GetBAngle(pcoord->x2v(j));
              Bx = std::cos(theta_b);
              Bz = std::sin(theta_b);
              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              // Bx = Bx * Bh/B_norm;
              // Bz = Bz * Bh/B_norm;
              Bx = Bx * Bmag/B_norm;
              Bz = Bz * Bmag/B_norm;
              }

            else{
              Bx = Bcx * Bmag/Bc;
              Bz = Bcz * Bmag/Bc;
            }
            By = 0;


            // Calculate normal-frame Lorentz factor
            Real uu1 = phydro->w(IVX,k,j,i);
            Real uu2 = phydro->w(IVY,k,j,i);
            Real uu3 = phydro->w(IVZ,k,j,i);
            Real tmp = g(I11,i) * SQR(uu1) + 2.0 * g(I12,i) * uu1 * uu2
                + 2.0 * g(I13,i) * uu1 * uu3 + g(I22,i) * SQR(uu2)
                + 2.0 * g(I23,i) * uu2 * uu3 + g(I33,i) * SQR(uu3);
            Real gamma = std::sqrt(1.0 + tmp);

            // Calculate 4-velocity
            Real alpha = std::sqrt(-1.0 / gi(I00,i));
            Real u0 = gamma / alpha;
            Real u1 = uu1 - alpha * gamma * gi(I01,i);
            Real u2 = uu2 - alpha * gamma * gi(I02,i);
            Real u3 = uu3 - alpha * gamma * gi(I03,i);

            Real u_0, u_1, u_2, u_3;

            u_0 = g(I00,i)*u0 + g(I01,i)*u1 + g(I02,i)*u2 + g(I03,i)*u3;
            u_1 = g(I01,i)*u0 + g(I11,i)*u1 + g(I12,i)*u2 + g(I13,i)*u3;
            u_2 = g(I02,i)*u0 + g(I12,i)*u1 + g(I22,i)*u2 + g(I23,i)*u3;
            u_3 = g(I03,i)*u0 + g(I13,i)*u1 + g(I23,i)*u2 + g(I33,i)*u3;
            // pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

            //Assume B^i_new = A_norm B^i
            //Then b^0 and b^i \propto A_norm 

            // Calculate 4-magnetic field
            Real bb1 = 0.0, bb2 = 0.0, bb3 = 0.0;
            Real b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
            Real b_0 = 0.0, b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
            bb1 = Bx;
            bb2 = By;
            bb3 = Bz;
            b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
            b1 = (bb1 + b0 * u1) / u0;
            b2 = (bb2 + b0 * u2) / u0;
            b3 = (bb3 + b0 * u3) / u0;
            // pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);


            b_0 = g(I00,i)*b0 + g(I01,i)*b1 + g(I02,i)*b2 + g(I03,i)*b3;
            b_1 = g(I01,i)*b0 + g(I11,i)*b1 + g(I12,i)*b2 + g(I13,i)*b3;
            b_2 = g(I02,i)*b0 + g(I12,i)*b1 + g(I22,i)*b2 + g(I23,i)*b3;
            b_3 = g(I03,i)*b0 + g(I13,i)*b1 + g(I23,i)*b2 + g(I33,i)*b3;
            
            // Calculate magnetic pressure
            Real b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;

            Bx = Bx * Bmag/std::sqrt(b_sq);
            By = By * Bmag/std::sqrt(b_sq);
            Bz = Bz * Bmag/std::sqrt(b_sq);


            pfield->b.x1f(k,j,i) = Bx;
          }
        }
      }
    
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju+1; j++) {
          for (int i=il; i<=iu; i++) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          pcoord->Face3Metric(k, j, il, iu, g, gi);
          for (int i=il; i<=iu; i++) {

            Real Bmag, exp_arg_term;

            if (pcoord->x2v(j) > 0.0){ // cold
              // exp_arg_term = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*y0 - SQR(shear_velocity);

              // // press = press_over_rho_interface*dc * std::pow( 1.0+ C_const/B_const *y, A_const/C_const);

              // Bmag = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const));
              // Bmag = Bc * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));


              Bmag =  std::sqrt(P_sol(j)/beta_c*2.0);

            }
            else{ // hot
              // exp_arg_term = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*y0;
              // // press = press_over_rho_interface*dc * std::pow( 1.0+ C_const/B_const *y, A_const/C_const);

              // Bmag = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x2v(j), A_const/C_const));

              // Bmag = Bh * std::sqrt( std::exp(pcoord->x2v(j)*exp_arg_term));

              Bmag =  std::sqrt(P_sol(j)/beta_h*2.0);
            }

            if (pcoord->x2v(j) < rotation_region_min){
              Bx = Bhx * Bmag/Bh;
              Bz = Bhz * Bmag/Bh;
            }
            else if (pcoord->x2v(j) < rotation_region_max){
              // Real w = ( pcoord->x2v(j) -rotation_region_min )/(rotation_region_max-rotation_region_min);

              // // Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_min);
              // // Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_min);
              // // Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_min);
              // // Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_min);


              // // Bx = Bhx_norm * std::sin((1.0-w)*theta_rot)/(sin(theta_rot) + 1e-10) + Bcx_norm*std::sin(w*theta_rot)/(sin(theta_rot) + 1e-10);
              // // Bz = Bhz_norm * std::sin((1.0-w)*theta_rot)/(sin(theta_rot) + 1e-10) + Bcz_norm*std::sin(w*theta_rot)/(sin(theta_rot) + 1e-10);


              // Bx = Bx_rotation_region_ymin + Bx_slope * ( pcoord->x2v(j) - rotation_region_min);
              // Bz = Bz_rotation_region_ymin + Bz_slope * ( pcoord->x2v(j) - rotation_region_min);

              Real theta_b = GetBAngle(pcoord->x2v(j));

              Bx = std::cos(theta_b);
              Bz = std::sin(theta_b);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
              // Bx = Bx * Bh/B_norm;
              // Bz = Bz * Bh/B_norm;
              Bx = Bx * Bmag/B_norm;
              Bz = Bz * Bmag/B_norm;
              }
            // else if (pcoord->x2v(j) < rotation_region_max){
            //   Bx = Bhx + Bx_slope * ( pcoord->x2v(j) - rotation_region_min);
            //   Bz = Bhz + Bz_slope * ( pcoord->x2v(j) - rotation_region_min);

            //   // Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x2v(j) - rotation_region_min);
            //   // Bz = Bhz_norm + Bz_slope_norm * ( pcoord->x2v(j) - rotation_region_min);

            //   //Now normalize

            //   Real B_norm = std::sqrt( SQR(Bx) + SQR(Bz) );
            //   // Bx = Bx * Bc/B_norm;
            //   // Bz = Bz * Bc/B_norm;
            //   Bx = Bx * Bmag/B_norm;
            //   Bz = Bz * Bmag/B_norm;
            // }
            else{
              Bx = Bcx * Bmag/Bc;
              Bz = Bcz * Bmag/Bc;
            }

            By = 0;

                        // Calculate normal-frame Lorentz factor
            Real uu1 = phydro->w(IVX,k,j,i);
            Real uu2 = phydro->w(IVY,k,j,i);
            Real uu3 = phydro->w(IVZ,k,j,i);
            Real tmp = g(I11,i) * SQR(uu1) + 2.0 * g(I12,i) * uu1 * uu2
                + 2.0 * g(I13,i) * uu1 * uu3 + g(I22,i) * SQR(uu2)
                + 2.0 * g(I23,i) * uu2 * uu3 + g(I33,i) * SQR(uu3);
            Real gamma = std::sqrt(1.0 + tmp);

            // Calculate 4-velocity
            Real alpha = std::sqrt(-1.0 / gi(I00,i));
            Real u0 = gamma / alpha;
            Real u1 = uu1 - alpha * gamma * gi(I01,i);
            Real u2 = uu2 - alpha * gamma * gi(I02,i);
            Real u3 = uu3 - alpha * gamma * gi(I03,i);

            Real u_0, u_1, u_2, u_3;
            // pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);


            u_0 = g(I00,i)*u0 + g(I01,i)*u1 + g(I02,i)*u2 + g(I03,i)*u3;
            u_1 = g(I01,i)*u0 + g(I11,i)*u1 + g(I12,i)*u2 + g(I13,i)*u3;
            u_2 = g(I02,i)*u0 + g(I12,i)*u1 + g(I22,i)*u2 + g(I23,i)*u3;
            u_3 = g(I03,i)*u0 + g(I13,i)*u1 + g(I23,i)*u2 + g(I33,i)*u3;

            //Assume B^i_new = A_norm B^i
            //Then b^0 and b^i \propto A_norm 

            // Calculate 4-magnetic field
            Real bb1 = 0.0, bb2 = 0.0, bb3 = 0.0;
            Real b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
            Real b_0 = 0.0, b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
            bb1 = Bx;
            bb2 = By;
            bb3 = Bz;
            b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
            b1 = (bb1 + b0 * u1) / u0;
            b2 = (bb2 + b0 * u2) / u0;
            b3 = (bb3 + b0 * u3) / u0;
            // pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

            b_0 = g(I00,i)*b0 + g(I01,i)*b1 + g(I02,i)*b2 + g(I03,i)*b3;
            b_1 = g(I01,i)*b0 + g(I11,i)*b1 + g(I12,i)*b2 + g(I13,i)*b3;
            b_2 = g(I02,i)*b0 + g(I12,i)*b1 + g(I22,i)*b2 + g(I23,i)*b3;
            b_3 = g(I03,i)*b0 + g(I13,i)*b1 + g(I23,i)*b2 + g(I33,i)*b3;
            
            
            // Calculate magnetic pressure
            Real b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;

            Bx = Bx * Bmag/std::sqrt(b_sq);
            By = By * Bmag/std::sqrt(b_sq);
            Bz = Bz * Bmag/std::sqrt(b_sq);


            // bb1 = Bx;
            // bb2 = By;
            // bb3 = Bz;
            // b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
            // b1 = (bb1 + b0 * u1) / u0;
            // b2 = (bb2 + b0 * u2) / u0;
            // b3 = (bb3 + b0 * u3) / u0;
            // // pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

            // b_0 = g(I00,i)*b0 + g(I01,i)*b1 + g(I02,i)*b2 + g(I03,i)*b3;
            // b_1 = g(I01,i)*b0 + g(I11,i)*b1 + g(I12,i)*b2 + g(I13,i)*b3;
            // b_2 = g(I02,i)*b0 + g(I12,i)*b1 + g(I22,i)*b2 + g(I23,i)*b3;
            // b_3 = g(I03,i)*b0 + g(I13,i)*b1 + g(I23,i)*b2 + g(I33,i)*b3;

            // b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;


            // fprintf(stderr,"bsq_target: %g bsq_act: %g\n", SQR(Bmag),b_sq);
            if (std::isnan(Bz)){
              Real udotu = u0*u_0 + u1*u_1 + u_2*u2 + u3*u_3;
              fprintf(stderr,"xyz: %g %g %g \n Bx: %g By: %g Bz: %g \n Bmag: %g b_sq: %g u0: %g\n bmu: %g %g %g %g \n udotu: %g\n",
                pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),Bx,By,Bz,Bmag,b_sq,
                      u0,b0,b1,b1,b3,udotu);
            }


            //Assume b^\mu = (b^0, A_norm Bx, A_norm By, A_norm Bz)
            //Use b^\mu b_\mu = Bmag^2 and b^\mu u^\nu g_\mu \nu =0
            // b^0 u^0 g_00 + A_norm( u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33) = 0
            // b^0 b^0 g_00 + A_norm^2 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33) = Bmag^2

            // solve for b^0 and A_norm

            // b^0 = - A_norm/(u^0 g_00) (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)
            // (b^0)^2 =  Bmag^2/g_00 -A_norm^2 /g_00 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33)  

            // A_norm^2 /(u^0 g_00)^2  (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)^2 = Bmag^2/g_00 -A_norm^2/g_00 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33)
            // A_norm^2 (1/(u^0)^2/g_00 (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)^2 +   (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33) = Bmag^2
            // A_norm = Bmag sqrt(1/ ... )

            // Real num_sq = SQR(Bmag) ;

            // Real denom_sq = 1.0/SQR(u0)/g(I00) * SQR(u1 * Bx * g(I11,i) + u2 * By * g(I22,i) + u3 * Bz * g(I33,i) )
            //                  + ( SQR(Bx) * g(I11,i) + SQR(By) * g(I22,i) + SQR(Bz) * g(I33,i) );


            // Real A_norm = std::sqrt(num_sq/denom_sq);

            // Real b0 = -A_norm / (u0*g(I00,i)) * ( u1 * Bx * g(I11,i) + u2 * By * g(I22,i) + u3 * Bz * g(I33,i) );

            // Real b1 = A_norm * Bx;
            // Real b2 = A_norm * By;
            // Real b3 = A_norm * Bz;

            //now convert back to three vector (Equation 17 Gammie+ 2003)


            // pfield->b.x3f(k,j,i) = b3 * u0 - b0 * u3;
            pfield->b.x3f(k,j,i) = Bz;
            pfield->b.x3f(k+1,j,i) = Bz;


            // if (std::isnan(pfield->b.x3f(k,j,i))){
            //   fprintf(stderr,"NAN in B3!!\n b3: %g u0: %g b0: %g u3: %g\n g: %g %g %g %g \n num_sq: %g denom_sq: %g A_norm: %g ",b3,u0,b0,u3, g(I00,i),g(I11,i),g(I22,i),g(I33,i),
            //                   num_sq,denom_sq,A_norm);
            // }
          }
        }
      }
      if (NON_BAROTROPIC_EOS) {
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            for (int i=il; i<=iu; i++) {
              // phydro->u(IEN,k,j,i) += 0.5*b0*b0;
            }
          }
        }
      }
      P_sol.DeleteAthenaArray();
    }

    // 3D PROBLEM ----------------------------------------------------------------

  } else {
    grav_acc = pin->GetReal("problem", "grav_acc");
    Real z0 = pin->GetReal("problem", "z0");


    AthenaArray<Real> P_sol;
    P_sol.NewAthenaArray(ku-kl+1);
    integrate_P_ODE(il,iu,jl,ju,kl,ku,pcoord->x3v,this,pin,P_sol );
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        pcoord->CellMetric(k, j, il, iu, g, gi);
        for (int i=il; i<=iu; i++) {

          // Real L = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
          Real den=1.0;
          Real dh = 1.0;
          Real dc = dh * drat;
          if (pcoord->x3v(k) > 0.0) den *= drat;


          Real C_const = -2.0*grav_acc;


          Real exp_arg_term,press,Bmag;

          den = P_sol(k)/press_over_rho_interface;
          // if (pcoord->x3v(k) > 0.0){ // cold
          //   exp_arg_term = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
          //   Real A_const = exp_arg_term;
          //   Real B_const = 1.0 + 2.0 * grav_acc*z0 - SQR(shear_velocity);
          //   press = press_over_rho_interface*dc * std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const);
          //   // press = press_over_rho_interface*dc * std::exp(pcoord->x3v(k)*exp_arg_term);

          //   den = dc * std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const);
          //   Bmag = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const));
          //   // den = dc * std::exp(pcoord->x3v(k)*exp_arg_term);
          //   // Bmag = Bc * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));

          // }
          // else{ // hot
          //   exp_arg_term = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
          //   Real A_const = exp_arg_term;
          //    Real B_const = 1.0 + 2.0 * grav_acc*z0;
          //   press = press_over_rho_interface*dh * std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const);
          //   // press = press_over_rho_interface*dh * std::exp(pcoord->x3v(k)*exp_arg_term);
          //   den = dh * std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const);
          //   Bmag = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const));

          //   // den = dh * std::exp(pcoord->x3v(k)*exp_arg_term);
          //   // Bmag = Bh * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));
          // }


          Real v3=0.0;


          phydro->w(IDN,k,j,i) = den;


          Real v1 = 0.0;
          v1 = shear_velocity/2.0;

          Real v2 = 0;


          Real u0 = std::sqrt( -1 / ( g(I00,i) + g(I11,i)*SQR(v1) + g(I22,i)*SQR(v2) + g(I33,i)*SQR(v3) + 
                                      2.0*g(I01,i)*v1 + 2.0*g(I02,i)*v2 + 2.0*g(I03,i)*v3  )   ); 
          Real u1 = u0*v1;
          Real u2 = u0*v2;
          Real u3 = u0*v3;

          // Real Lorentz = 1.0/std::sqrt(-g(I00,i) - SQR(v2));
          // Real u0 = Lorentz;
          // Real u1 = 0.0;
          // Real u2 = v2*Lorentz;
          // Real u3 = 0.0;

          // Now convert to Athena++ velocities (see White+ 2016)
          Real uu1 = u1 - gi(I01,i) / gi(I00,i) * u0;
          Real uu2 = u2 - gi(I02,i) / gi(I00,i) * u0;
          Real uu3 = u3 - gi(I03,i) / gi(I00,i) * u0;

          // Real Lorentz = 1.0/std::sqrt(-g(I00,i) - SQR(v3));

          // // Real Lorentz = 1.0/std::sqrt(-g(I00,i) - SQR(v2));
          // Real u0 = Lorentz;
          // Real u1 = 0.0;
          // Real u2 = 0.0;
          // Real u3 = v3 * Lorentz;

          
          phydro->w(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = uu3;


          if (NON_BAROTROPIC_EOS) {
            phydro->w(IPR,k,j,i) =  P_sol(k);

            // phydro->w(IPR,k,j,i) =  press_over_rho_interface*den + grav_acc*den*(pcoord->x3v(k));
          }
        }
      }
    }

    // initialize interface B
    if (MAGNETIC_FIELDS_ENABLED) {
      // Read magnetic field strength, angle [in degrees, 0 is along +ve X-axis]



      // Real rotation_region_min = 3.0*L/8.0 +  pmy_mesh->mesh_size.x3min;
      // Real rotation_region_max = rotation_region_min + L/4.0;

      Real Bin = ( Bh * Bc * std::sin(theta_rot) ) / std::sqrt( SQR(Bh) + SQR(Bc) + 2.0*Bh*Bc*std::cos(theta_rot) ) ;
      Real Bhx = Bin;
      Real Bcx = - Bhx;

      Real sign_flip = 1.0;
      if (std::cos(theta_rot)<0.0) sign_flip=-1.0;
      Real Bhy = sign_flip * std::sqrt( SQR(Bh) - SQR(Bin) );
      Real Bcy =             std::sqrt( SQR(Bc) - SQR(Bin) );

      // Real Bx_slope = (Bcx - Bhx) / ( length_of_rotation_region) ; 
      // Real By_slope = (Bcy - Bhy) / ( length_of_rotation_region) ; 



      // Real Bhx_norm = Bhx/Bh;
      // Real Bcx_norm = Bcx/Bc;
      // Real Bhy_norm = Bhy/Bh;
      // Real Bcy_norm = Bcy/Bc;


      // I DON"T THINK THIS MAKES SENSE. Rotate angle, not linearly
      // Real Bx_slope_norm = (Bcx_norm - Bhx_norm) / ( length_of_rotation_region) ; 
      // Real By_slope_norm = (Bcy_norm - Bhy_norm) / ( length_of_rotation_region) ;

      // Real C_const = -2.0*grav_acc;


      // Real exp_arg_term_rotation_region_zmax = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
      // Real A_const_rotation_region_zmax = exp_arg_term_rotation_region_zmax;
      // Real B_const_rotation_region_zmax = 1.0 + 2.0 * grav_acc*z0 - SQR(shear_velocity);

      // Real Bmag_rotation_region_zmax = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const_rotation_region_zmax *rotation_region_max, A_const_rotation_region_zmax/C_const));

      // Real Bx_rotation_region_zmax = Bcx * Bmag_rotation_region_zmax/Bc;
      // Real By_rotation_region_zmax = Bcy * Bmag_rotation_region_zmax/Bc;


      // Real exp_arg_term_rotation_region_zmin = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
      // Real A_const_rotation_region_zmin = exp_arg_term_rotation_region_zmin;
      // Real B_const_rotation_region_zmin = 1.0 + 2.0 * grav_acc*z0;

      // Real Bmag_rotation_region_zmin = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const_rotation_region_zmin *rotation_region_min, A_const_rotation_region_zmin/C_const));

      // Real Bx_rotation_region_zmin = Bhx * Bmag_rotation_region_zmin/Bh;
      // Real By_rotation_region_zmin = Bhy * Bmag_rotation_region_zmin/Bh;


      // Real Bx_slope = (Bx_rotation_region_zmax - Bx_rotation_region_zmin) / ( length_of_rotation_region) ; 
      // Real By_slope = (By_rotation_region_zmax - By_rotation_region_zmin) / ( length_of_rotation_region) ; 


      Real Bx, By,Bz;
      // angle = (angle/180.)*PI;

      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          pcoord->Face1Metric(k, j, il, iu+1, g, gi);
          for (int i=il; i<=iu+1; i++) {

            Real exp_arg_term,Bmag;


            if (pcoord->x3v(k) > 0.0){ // cold
              // exp_arg_term = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*z0 - SQR(shear_velocity);

              // Bmag = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const));

              Bmag = std::sqrt( P_sol(k) * (1/beta_c)*2.0 );

              // Bmag = Bc * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));

            }
            else{ // hot
              // exp_arg_term = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              // // Bmag = Bh * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*z0;

              // Bmag = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const));

              Bmag = std::sqrt( P_sol(k) * (1/beta_h)*2.0 );

            }


            if (pcoord->x3v(k) < rotation_region_min){
              Bx = Bhx * Bmag/Bh;
              By = Bhy * Bmag/Bh;
            }
            else if (pcoord->x3v(k) < rotation_region_max){
              // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_min);
              // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_min);


              // Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_min);
              // By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_min);

              // Bx = Bx_rotation_region_zmin + Bx_slope * ( pcoord->x3v(k) - rotation_region_min);
              // By = By_rotation_region_zmin + By_slope * ( pcoord->x3v(k) - rotation_region_min);

              Real theta_b = GetBAngle(pcoord->x3v(k));

              Bx = std::cos(theta_b);
              By = std::sin(theta_b);

              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bmag/B_norm;
              By = By * Bmag/B_norm;
              }
            // else if (pcoord->x3v(k) < rotation_region_max){
            //   // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_min);
            //   // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_min);

            //   Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_min);
            //   By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_min);

            //   //Now normalize

            //   Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
            //   Bx = Bx * Bmag/B_norm;
            //   By = By * Bmag/B_norm;
            // }
            else{
              Bx = Bcx * Bmag/Bc;
              By = Bcy * Bmag/Bc;
            }

            Bz = 0.0;

            // Calculate normal-frame Lorentz factor
            Real uu1 = phydro->w(IVX,k,j,i);
            Real uu2 = phydro->w(IVY,k,j,i);
            Real uu3 = phydro->w(IVZ,k,j,i);
            Real tmp = g(I11,i) * SQR(uu1) + 2.0 * g(I12,i) * uu1 * uu2
                + 2.0 * g(I13,i) * uu1 * uu3 + g(I22,i) * SQR(uu2)
                + 2.0 * g(I23,i) * uu2 * uu3 + g(I33,i) * SQR(uu3);
            Real gamma = std::sqrt(1.0 + tmp);

            // Calculate 4-velocity
            Real alpha = std::sqrt(-1.0 / gi(I00,i));
            Real u0 = gamma / alpha;
            Real u1 = uu1 - alpha * gamma * gi(I01,i);
            Real u2 = uu2 - alpha * gamma * gi(I02,i);
            Real u3 = uu3 - alpha * gamma * gi(I03,i);

            Real u_0, u_1, u_2, u_3;

            u_0 = g(I00,i)*u0 + g(I01,i)*u1 + g(I02,i)*u2 + g(I03,i)*u3;
            u_1 = g(I01,i)*u0 + g(I11,i)*u1 + g(I12,i)*u2 + g(I13,i)*u3;
            u_2 = g(I02,i)*u0 + g(I12,i)*u1 + g(I22,i)*u2 + g(I23,i)*u3;
            u_3 = g(I03,i)*u0 + g(I13,i)*u1 + g(I23,i)*u2 + g(I33,i)*u3;
            //Assume B^i_new = A_norm B^i
            //Then b^0 and b^i \propto A_norm 

            // Calculate 4-magnetic field
            Real bb1 = 0.0, bb2 = 0.0, bb3 = 0.0;
            Real b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
            Real b_0 = 0.0, b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
            bb1 = Bx;
            bb2 = By;
            bb3 = Bz;
            b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
            b1 = (bb1 + b0 * u1) / u0;
            b2 = (bb2 + b0 * u2) / u0;
            b3 = (bb3 + b0 * u3) / u0;

            b_0 = g(I00,i)*b0 + g(I01,i)*b1 + g(I02,i)*b2 + g(I03,i)*b3;
            b_1 = g(I01,i)*b0 + g(I11,i)*b1 + g(I12,i)*b2 + g(I13,i)*b3;
            b_2 = g(I02,i)*b0 + g(I12,i)*b1 + g(I22,i)*b2 + g(I23,i)*b3;
            b_3 = g(I03,i)*b0 + g(I13,i)*b1 + g(I23,i)*b2 + g(I33,i)*b3;
                        
            // Calculate magnetic pressure
            Real b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;

            Bx = Bx * Bmag/std::sqrt(b_sq);
            By = By * Bmag/std::sqrt(b_sq);
            Bz = Bz * Bmag/std::sqrt(b_sq);



            //Assume b^\mu = (b^0, A_norm Bx, A_norm By, A_norm Bz)
            //Use b^\mu b_\mu = Bmag^2 and b^\mu u^\nu g_\mu \nu =0
            // b^0 u^0 g_00 + A_norm( u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33) = 0
            // b^0 b^0 g_00 + A_norm^2 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33) = Bmag^2

            // solve for b^0 and A_norm

            // b^0 = - A_norm/(u^0 g_00) (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)
            // (b^0)^2 =  Bmag^2/g_00 -A_norm^2 /g_00 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33)  

            // A_norm^2 /(u^0 g_00)^2  (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)^2 = Bmag^2/g_00 -A_norm^2/g_00 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33)
            // A_norm^2 (1/(u^0)^2/g_00 (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)^2 +   (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33) = Bmag^2
            // A_norm = Bmag sqrt(1/ ... )

            // Real num_sq = SQR(Bmag) ;

            // Real denom_sq = 1.0/SQR(u0)/g(I00) * SQR(u1 * Bx * g(I11,i) + u2 * By * g(I22,i) + u3 * Bz * g(I33,i) )
            //                  + ( SQR(Bx) * g(I11,i) + SQR(By) * g(I22,i) + SQR(Bz) * g(I33,i) );


            // Real A_norm = std::sqrt(num_sq/denom_sq);

            // Real b0 = -A_norm / (u0*g(I00,i)) * ( u1 * Bx * g(I11,i) + u2 * By * g(I22,i) + u3 * Bz * g(I33,i) );

            // Real b1 = A_norm * Bx;
            // Real b2 = A_norm * By;
            // Real b3 = A_norm * Bz;

            //now convert back to three vector (Equation 17 Gammie+ 2003)
   
            // pfield->b.x1f(k,j,i) = b1 * u0 - b0 * u1;
            pfield->b.x1f(k,j,i) = Bx;
          }
        }
      }
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju+1; j++) {
          pcoord->Face2Metric(k, j, il, iu+1, g, gi);
          for (int i=il; i<=iu; i++) {


            Real exp_arg_term,Bmag;
            Real C_const = -2.0*grav_acc;

            if (pcoord->x3v(k) > 0.0){ // cold
              // exp_arg_term = grav_acc / sigma_c * (2.0 + gamma_adi/gm1*sigma_c*beta_c + 2.0*sigma_c) / (1.0 + beta_c);
              // Real A_const = exp_arg_term;
              // Real B_const = 1.0 + 2.0 * grav_acc*z0 - SQR(shear_velocity);

              // Bmag = Bc * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const));

              Bmag = std::sqrt(P_sol(k)/beta_c*2.0);

              // Bmag = Bc * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));

            }
            else{ // hot
              // exp_arg_term = grav_acc / sigma_h * (2.0 + gamma_adi/gm1*sigma_h*beta_h + 2.0*sigma_h) / (1.0 + beta_h);
              // // Bmag = Bh * std::sqrt( std::exp(pcoord->x3v(k)*exp_arg_term));
              // Real A_const = exp_arg_term;
              // Real B_const =  1.0 + 2.0 * grav_acc*z0;

              // Bmag = Bh * std::sqrt( std::pow( 1.0+ C_const/B_const *pcoord->x3v(k), A_const/C_const));

              Bmag = std::sqrt(P_sol(k)/beta_h*2.0);
            }


            if (pcoord->x3v(k) < rotation_region_min){
              Bx = Bhx * Bmag/Bh;
              By = Bhy * Bmag/Bh;
            }
            else if (pcoord->x3v(k) < rotation_region_max){
              // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_min);
              // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_min);

              // Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_min);
              // By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_min);

              // Bx = Bx_rotation_region_zmin + Bx_slope * ( pcoord->x3v(k) - rotation_region_min);
              // By = By_rotation_region_zmin + By_slope * ( pcoord->x3v(k) - rotation_region_min);


              Real theta_b = GetBAngle(pcoord->x3v(k));

              Bx = std::cos(theta_b);
              By = std::sin(theta_b);


              //Now normalize

              Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
              Bx = Bx * Bmag/B_norm;
              By = By * Bmag/B_norm;
              }
            // else if (pcoord->x3v(k) < rotation_region_max){
            //   // Bx = Bhx + Bx_slope * ( pcoord->x3v(k) - rotation_region_min);
            //   // By = Bhy + By_slope * ( pcoord->x3v(k) - rotation_region_min);

            //   Bx = Bhx_norm + Bx_slope_norm * ( pcoord->x3v(k) - rotation_region_min);
            //   By = Bhy_norm + By_slope_norm * ( pcoord->x3v(k) - rotation_region_min);

            //   //Now normalize

            //   Real B_norm = std::sqrt( SQR(Bx) + SQR(By) );
            //   Bx = Bx * Bmag/B_norm;
            //   By = By * Bmag/B_norm;
            // }
            else{
              Bx = Bcx * Bmag/Bc;
              By = Bcy * Bmag/Bc;
            }

            Bz = 0.0;


            // Calculate normal-frame Lorentz factor
            Real uu1 = phydro->w(IVX,k,j,i);
            Real uu2 = phydro->w(IVY,k,j,i);
            Real uu3 = phydro->w(IVZ,k,j,i);
            Real tmp = g(I11,i) * SQR(uu1) + 2.0 * g(I12,i) * uu1 * uu2
                + 2.0 * g(I13,i) * uu1 * uu3 + g(I22,i) * SQR(uu2)
                + 2.0 * g(I23,i) * uu2 * uu3 + g(I33,i) * SQR(uu3);
            Real gamma = std::sqrt(1.0 + tmp);

            // Calculate 4-velocity
            Real alpha = std::sqrt(-1.0 / gi(I00,i));
            Real u0 = gamma / alpha;
            Real u1 = uu1 - alpha * gamma * gi(I01,i);
            Real u2 = uu2 - alpha * gamma * gi(I02,i);
            Real u3 = uu3 - alpha * gamma * gi(I03,i);
            Real u_0, u_1, u_2, u_3;

            u_0 = g(I00,i)*u0 + g(I01,i)*u1 + g(I02,i)*u2 + g(I03,i)*u3;
            u_1 = g(I01,i)*u0 + g(I11,i)*u1 + g(I12,i)*u2 + g(I13,i)*u3;
            u_2 = g(I02,i)*u0 + g(I12,i)*u1 + g(I22,i)*u2 + g(I23,i)*u3;
            u_3 = g(I03,i)*u0 + g(I13,i)*u1 + g(I23,i)*u2 + g(I33,i)*u3;
            //Assume B^i_new = A_norm B^i
            //Then b^0 and b^i \propto A_norm 

            // Calculate 4-magnetic field
            Real bb1 = 0.0, bb2 = 0.0, bb3 = 0.0;
            Real b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
            Real b_0 = 0.0, b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
            bb1 = Bx;
            bb2 = By;
            bb3 = Bz;
            b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3;
            b1 = (bb1 + b0 * u1) / u0;
            b2 = (bb2 + b0 * u2) / u0;
            b3 = (bb3 + b0 * u3) / u0;

            b_0 = g(I00,i)*b0 + g(I01,i)*b1 + g(I02,i)*b2 + g(I03,i)*b3;
            b_1 = g(I01,i)*b0 + g(I11,i)*b1 + g(I12,i)*b2 + g(I13,i)*b3;
            b_2 = g(I02,i)*b0 + g(I12,i)*b1 + g(I22,i)*b2 + g(I23,i)*b3;
            b_3 = g(I03,i)*b0 + g(I13,i)*b1 + g(I23,i)*b2 + g(I33,i)*b3;
                        
            // Calculate magnetic pressure
            Real b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;

            Bx = Bx * Bmag/std::sqrt(b_sq);
            By = By * Bmag/std::sqrt(b_sq);
            Bz = Bz * Bmag/std::sqrt(b_sq);


            //Assume b^\mu = (b^0, A_norm Bx, A_norm By, A_norm Bz)
            //Use b^\mu b_\mu = Bmag^2 and b^\mu u^\nu g_\mu \nu =0
            // b^0 u^0 g_00 + A_norm( u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33) = 0
            // b^0 b^0 g_00 + A_norm^2 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33) = Bmag^2

            // solve for b^0 and A_norm

            // b^0 = - A_norm/(u^0 g_00) (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)
            // (b^0)^2 =  Bmag^2/g_00 -A_norm^2 /g_00 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33)  
            // A_norm^2 /(u^0 g_00)^2  (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)^2 = Bmag^2/g_00 -A_norm^2/g_00 (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33)
            // A_norm^2 (1/(u^0)^2/g_00 (u^1 Bx g_11 + u^2 By g_22 + u^3 Bz g_33)^2 +   (Bx^2 g_11 + By^2 g_22 + Bz^2 g_33) = Bmag^2
            // A_norm = Bmag sqrt(1/ ... )

            // Real num_sq = SQR(Bmag) ;

            // Real denom_sq = 1.0/SQR(u0)/g(I00) * SQR(u1 * Bx * g(I11,i) + u2 * By * g(I22,i) + u3 * Bz * g(I33,i) )
            //                  + ( SQR(Bx) * g(I11,i) + SQR(By) * g(I22,i) + SQR(Bz) * g(I33,i) );


            // Real A_norm = std::sqrt(num_sq/denom_sq);

            // Real b0 = -A_norm / (u0*g(I00,i)) * ( u1 * Bx * g(I11,i) + u2 * By * g(I22,i) + u3 * Bz * g(I33,i) );

            // Real b1 = A_norm * Bx;
            // Real b2 = A_norm * By;
            // Real b3 = A_norm * Bz;

            //now convert back to three vector (Equation 17 Gammie+ 2003)



            // pfield->b.x2f(k,j,i) = b2 * u0 - b0 * u2;
            pfield->b.x2f(k,j,i) = By;

          }
        }
      }
      for (int k=kl; k<=ku+1; k++) {
        for (int j=jl; j<=ju; j++) {
          for (int i=il; i<=iu; i++) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
      if (NON_BAROTROPIC_EOS) {
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            for (int i=il; i<=iu; i++) {
              // phydro->u(IEN,k,j,i) += 0.5*b0*b0;
            }
          }
        }
      }
    }

    P_sol.DeleteAthenaArray();
  } // end of 3D initialization



    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        pcoord->CellMetric(k, j, il, iu, g, gi);
        for (int i=il; i<=iu; i++) {

                        // Calculate normal-frame Lorentz factor
            Real uu1 = phydro->w(IVX,k,j,i);
            Real uu2 = phydro->w(IVY,k,j,i);
            Real uu3 = phydro->w(IVZ,k,j,i);
            Real tmp = g(I11,i) * SQR(uu1) + 2.0 * g(I12,i) * uu1 * uu2
                + 2.0 * g(I13,i) * uu1 * uu3 + g(I22,i) * SQR(uu2)
                + 2.0 * g(I23,i) * uu2 * uu3 + g(I33,i) * SQR(uu3);
            Real gamma = std::sqrt(1.0 + tmp);

            // Calculate 4-velocity
            Real alpha = std::sqrt(-1.0 / gi(I00,i));
            Real u0 = gamma / alpha;
            Real u1 = uu1 - alpha * gamma * gi(I01,i);
            Real u2 = uu2 - alpha * gamma * gi(I02,i);
            Real u3 = uu3 - alpha * gamma * gi(I03,i);

            Real v1 = u1/u0;
            Real v2 = u2/u0;
            Real v3 = u3/u0;

            Real rand_number;

            if (block_size.nx3 == 1) { //2D
                if (iprob == 1) {
                  v2 += amp*cs*(1.0 + std::cos(kx*pcoord->x1v(i)))*
                                         (1.0 + std::cos(ky*pcoord->x2v(j)))/4.0;
                } else {
                  rand_number  = ran2(&iseed);
                 v2 += amp*cs* (rand_number - 0.5)*(1.0+std::cos(ky*pcoord->x2v(j)));
                }
            }

            else{  //3D
              if (iprob == 1) {
                v3 += amp*cs*(1.0 + std::cos(kx*(pcoord->x1v(i))))/8.0
                                       *(1.0 + std::cos(ky*pcoord->x2v(j)))
                                       *(1.0 + std::cos(kz*pcoord->x3v(k)));
              } else {
                v3 += amp*cs*(ran2(&iseed) - 0.5)*(
                    1.0 + std::cos(kz*pcoord->x3v(k)));
              }
            }
      
            u0 = std::sqrt( -1 / ( g(I00,i) + g(I11,i)*SQR(v1) + g(I22,i)*SQR(v2) + g(I33,i)*SQR(v3) + 
                                        2.0*g(I01,i)*v1 + 2.0*g(I02,i)*v2 + 2.0*g(I03,i)*v3  )   ); 
            u1 = u0*v1;
            u2 = u0*v2;
            u3 = u0*v3;

            // Now convert to Athena++ velocities (see White+ 2016)
            uu1 = u1 - gi(I01,i) / gi(I00,i) * u0;
            uu2 = u2 - gi(I02,i) / gi(I00,i) * u0;
            uu3 = u3 - gi(I03,i) / gi(I00,i) * u0;

            phydro->w(IM1,k,j,i) = uu1;
            phydro->w(IM2,k,j,i) = uu2;
            phydro->w(IM3,k,j,i) = uu3;


          }
        }
      }


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
    

    UserWorkInLoop();
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


    // Do nothing

    // copy face-centered magnetic fields into ghost zones, reflecting b2
  if (MAGNETIC_FIELDS_ENABLED) {

    for (int k=kl; k<=ku+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
         fprintf(stderr,"B3 in boundary: %g i j k : %d %d %d  \n ", b.x3f(k,(jl-j),i), i, jl-j, k );
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


  // do nothing
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

// do nothing

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
// do nothing

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

  Real Phi = Phi_func(z,grav_acc,z0); 

  // Set covariant components
  g(I00) = -(1.0 +2.0*Phi);
  g(I01) = 0;
  g(I02) = 0;
  g(I03) = -2.0 * Phi;
  g(I11) = 1.0;
  g(I12) = 0;
  g(I13) = 0;
  g(I22) = 1.0 ;
  g(I23) = 0;
  g(I33) = 1.0 - 2.0 * Phi ;




  // // Set contravariant components
  g_inv(I00) = -g(I33);
  g_inv(I01) = 0;
  g_inv(I02) = 0;
  g_inv(I03) = g(I03);
  g_inv(I11) = 1.0;
  g_inv(I12) = 0;
  g_inv(I13) = 0;
  g_inv(I22) = 1.0;
  g_inv(I23) = 0.0;
  g_inv(I33) = -g(I00);




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
  dg_dx3(I03) = 2.0 * grav_acc;
  dg_dx3(I11) = 0;
  dg_dx3(I12) = 0;
  dg_dx3(I13) = 0;
  dg_dx3(I22) = 0;
  dg_dx3(I23) = 0;
  dg_dx3(I33) = 2.0 * grav_acc;



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
  Real Phi = Phi_func(y,grav_acc,y0);  // -grav_acc*(y-y0);
  // Set covariant components
  g(I00) = -(1.0 + 2.0*Phi);
  g(I01) = 0;
  g(I02) = -2.0*Phi;
  g(I03) = 0;
  g(I11) = 1.0;
  g(I12) = 0;
  g(I13) = 0;
  g(I22) = 1.0 - 2.0 * Phi;
  g(I23) = 0;
  g(I33) = 1.0 ;




  // // Set contravariant components
  g_inv(I00) = -g(I22);
  g_inv(I01) = 0;
  g_inv(I02) = g(I02);
  g_inv(I03) = 0.0;
  g_inv(I11) = 1.0;
  g_inv(I12) = 0;
  g_inv(I13) = 0;
  g_inv(I22) = -g(I00);
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
  dg_dx2(I02) = 2.0 * grav_acc;
  dg_dx2(I03) = 0;
  dg_dx2(I11) = 0;
  dg_dx2(I12) = 0;
  dg_dx2(I13) = 0;
  dg_dx2(I22) = 2.0 * grav_acc;
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



