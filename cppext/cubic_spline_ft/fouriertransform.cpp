/*****************************************************************************
 *
 * ALPS DMFT Project
 *
 * Copyright (C) 2005 - 2009 by Emanuel Gull <gull@phys.columbia.edu>
 *                              Philipp Werner <werner@itp.phys.ethz.ch>,
 *                              Sebastian Fuchs <fuchs@theorie.physik.uni-goettingen.de>
 *                              Matthias Troyer <troyer@comp-phys.org>
 *
 *
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/


#include <boost/numeric/bindings/ublas.hpp>
#include <boost/numeric/bindings/lapack/driver/gesv.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <math.h>
#include "fouriertransform.h"

using namespace Eigen;

typedef boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> dense_matrix;

FourierTransformer::FourierTransformer(const double beta, 
                                       const VectorXd &ftail)
    : beta_(beta), c_(ftail) {
  if (c_.size() < 3) 
    throw std::length_error("The tail should contains up to at least the third order");
}

// backward_ft G(iwn)->G(tau) works for cluster Green's functions
void FourierTransformer::backward_ft(const VectorXcd &G_omega, 
                                     VectorXd &G_tau, const int N_tau) const {
  unsigned int N_omega = G_omega.size();
  VectorXcd G_omega_no_model(G_omega);
  double dt = beta_/N_tau;
  
  G_tau.resize(N_tau+1);
  if (c_(0) == 0 && c_(1) == 0 && c_(2) == 0) {
    //nothing happening in this gf.
    for (int i = 0; i <= N_tau; i++) {
      G_tau(i)=0.;
    }
  } else {
    for (int k = 0; k < N_omega; k++) {
      std::complex<double> iw(0,(2*k+1)*M_PI/beta_);
      G_omega_no_model(k) -= f_omega(iw, c_(0),c_(1), c_(2));
    }
    for (int i=0; i<N_tau; i++) {
      G_tau(i) = f_tau(i*dt, beta_, c_(0), c_(1), c_(2));
      for (int k=0; k<N_omega; k++) {
        double wt((2*k+1)*i*M_PI/N_tau);
        G_tau(i) += 2/beta_*(cos(wt)*G_omega_no_model(k).real()+
                             sin(wt)*G_omega_no_model(k).imag());
      }
    }
    G_tau(N_tau) = -c_[0];
    G_tau(N_tau)-= G_tau(0);
  }
}

void generate_spline_matrix(dense_matrix & spline_matrix, double dt) {
  // spline_matrix has dimension (N+1)x(N+1)
  int Np1 = spline_matrix.size1();
  //std::cout<<"spline matrix size is: "<<Np1<<std::endl;
  // A is the matrix whose inverse defines spline_matrix
  //   
  //      6                   6
  //      1  4  1
  //         1  4  1
  // A =        ...
  //
  //                    1  4  1
  //     -2 -1     0       1  2
  spline_matrix.clear(); 
  dense_matrix A = 4*dt/6.*boost::numeric::ublas::identity_matrix<double>(Np1);
  
  for (int i=1; i<Np1-1; i++) {
    A(i,i-1) = dt/6.;
    A(i,i+1) = dt/6.;
  }
  A(0,0) = 1.;
  A(0, Np1-1) = 1.;
  A(Np1-1, 0) = -2.*dt/6.;
  A(Np1-1, 1) = -1.*dt/6.;
  A(Np1-1, Np1-2) = 1*dt/6.;
  A(Np1-1, Np1-1) = 2*dt/6.;
  
  // solve A*spline_matrix=I
  // gesv solves A*X=B, input for B is I, output (=solution X) is spline_matrix
  spline_matrix = boost::numeric::ublas::identity_matrix<double>(Np1);   
  boost::numeric::ublas::vector<fortran_int_t> ipivot(A.size1());
  boost::numeric::bindings::lapack::gesv(A, ipivot,spline_matrix);
}

void evaluate_second_derivatives(double dt, dense_matrix & spline_matrix, 
    std::vector<double> & g, std::vector<double> & second_derivatives, 
    const double c1g, const double c2g, const double c3g) {
  // g, rhs and second_derivatives have dimension N+1
  int Np1 = spline_matrix.size1();
  //assert(c1g==1); 
  // rhs is the vector containing the data of the curve y = g(tau), which allows to 
  // compute the vector of second derivatives y'' at times tau_n by evaluating
  // y'' = spline_matrix * rhs(y)
  //
  //                         0                                
  //                         y0 - 2*y1 + y2
  //                         y1 - 2*y2 + y3
  // rhs = 6/(delta_tau)^2 * ...
  //
  //                         yNp1-3 - 2*yNp1-2 + yNp1-1
  //                         y0 - y1 + yNp1-2 - yNp1-1     
  
  std::vector<double> rhs(Np1, 0);
  std::cout<<"constants: "<<c1g<<" "<<c2g<<" "<<c3g<<std::endl;
  rhs[0] = -c3g; //G''(0)+G''(beta)=-c3
  for (int i=1; i<Np1-1; i++) {
    rhs[i] = (g[i-1]-2*g[i]+g[i+1])/dt;
  }
  rhs[Np1-1] = c2g -1./dt*(-g[0] + g[1] -g[Np1-2] + g[Np1-1]);
  
  for (int i=0; i<Np1; i++) {
    second_derivatives[i]=0;
    for (int j=0; j<Np1; j++) {
      second_derivatives[i] += spline_matrix(i,j)*rhs[j];
    }
  }
}

// 
// calculations following Armin Comadac's thesis (see the thesis for details)
// S1 = G'(0) + G'(beta) = y'(0) + y'(L)
// S2 = G''(0) + G''(beta) = y''(0) + y''(L)
// remember that S1 and S2 can be different from the Green's function's tail
// in details: S1 = G2
//             S2 = -G3
//
VectorXd ArminGetSecondDerivative(const VectorXd &y, 
    const double beta, const double S1, const double S2) {
  const int L = y.size()-1;
  double dx = beta/L;

  VectorXd X(L+1), Y(L), Z(L);
  // coefficients for L, U matrices (Armin's notations)
  VectorXd a(L), b(L), d(L);
  Y.setZero();
  Z.setZero();
  Y(0) = 6. * ((y[1]-y[0]+y[L]-y[L-1]-S1*dx) / (dx*dx) + S2/3.);
  Z(0) = Y(0);
  a(0) = 4.;
  b(0) = -1./a(0);
  d(0) = -1.;
  for (int i = 1; i < L-1; ++i) {
    a(i) = 4. - 1./a(i-1);
    b(i) = -b(i-1)/a(i);
    d(i) = -d(i-1)/a(i-1);
    Y(i) = 6. * (y(i+1) - 2*y(i) + y(i-1)) / (dx*dx);
    Z(i) = Y(i) - Z(i-1)/a(i-1);
  }
  Y(L-1) = 6. * ((y(L) - 2*y(L-1) + y(L-2)) / (dx*dx) - S2 / 6.);
  a(L-1) = 4.;
  b(L-2) = (1.-b(L-3))/a(L-2);
  d(L-2) = 1. - d(L-3)/a(L-3);
  for (int i = 0; i < L-1; ++i)
    a(L-1) -= b(i)*d(i);
  Z(L-1) = Y(L-1);
  for (int i = 0; i < L-1; ++i)
    Z(L-1) -= b(i)*Z(i);

  X(L-1) = Z(L-1) / a(L-1);
  X(L-2) = (Z(L-2)-d(L-2)*X(L-1)) / a(L-2);
  for (int i = L-3; i >= 0; --i)
    X(i) = (Z(i) - X(i+1) - d(i)*X(L-1))/a(i);
  X(L) = S2 - X(0);

  return X;
}


// forward_ft G(tau)->G(iwn) works for cluster Green's functions
void FourierTransformer::Armin_forward_ft(const VectorXd &gtau, VectorXcd &gomega,
                                          const int num_matsubara) const {
  int N = gtau.size()-1;
  double dt = beta_/N;
  VectorXcd v_omega(num_matsubara);
  gomega.resize(num_matsubara);
  VectorXd v2(ArminGetSecondDerivative(gtau, beta_, c_(1), -c_(2)));

  // DEBUG
//  const VectorXd &v(gtau);
//  std::cout<<"c1 is: "<<c_(0)<<" computed: "<<-v[0]-v[N]<<std::endl;
//  std::cout<<"c2 is: "<<c_(1)<<" computed: "<<(v[1]-v[0]+v[N]-v[N-1])/dt<<std::endl;
//  std::cout<<"c3 is: "<<c_(2)<<" computed: "<<-(v[2]-2*v[1]+v[0]+v[N-2]-2*v[N-1]+v[N])/(dt*dt)<<std::endl;

  v_omega.setZero();
  for (int k = 0; k < num_matsubara; ++k) {
    std::complex<double> iw(0, M_PI*(2*k+1)/beta_);
    for (int n = 1; n < N; n++) {
      //partial integration, four times. 
      //Then approximate the fourth derivative by finite differences
      v_omega(k) += exp(iw*(n*dt))*(v2(n+1)-2*v2(n)+v2(n-1)); 
    }
    // the third derivative, on the boundary
    v_omega(k) += (v2(1) - v2(0) + v2(N) - v2(N-1)); 
    v_omega(k) *= 1./(dt*iw*iw*iw*iw);
    // the boundary terms of the first, second, and third partial integration.
    v_omega(k) += f_omega(iw, c_(0), c_(1), c_(2)); 
    //std::cout<<"derivative at boundary: "<<-v2[1] + v2[0] + v2[N] - v2[N-1]<<" divisor: "<< 1./(dt*iw*iw*iw*iw)<<std::endl;
    // the proper convention for the self consistency loop here.
    gomega(k)=v_omega(k); 
  }
} 

// forward_ft G(tau)->G(iwn) works for cluster Green's functions
void FourierTransformer::forward_ft(const VectorXd &gtau, VectorXcd &gomega,
                                    const int num_matsubara) const {
  std::vector<double> v(gtau.size());
  std::vector<std::complex<double> > v_omega(num_matsubara);
  int Np1 = v.size();
  int N = Np1-1;
  int N_omega = v_omega.size();
  double dt = beta_/N;

  gomega.resize(num_matsubara);
  for(int tau=0;tau<Np1;++tau) {
    v[tau]=gtau(tau);
  }
  
  dense_matrix spline_matrix(Np1, Np1);
  generate_spline_matrix(spline_matrix, dt);
  // matrix containing the second derivatives y'' 
  // of interpolated y=v[tau] at points tau_n 
  std::vector<double> v2(Np1, 0); 
  evaluate_second_derivatives(dt/*,beta_*/, spline_matrix, v, v2, 
      c_(0), c_(1), c_(2));
  v_omega.assign(N_omega, 0);
  for (int k=0; k<N_omega; k++) {
    std::complex<double> iw(0, M_PI*(2*k+1)/beta_);
    for (int n=1; n<N; n++) {
      //partial integration, four times. 
      //Then approximate the fourth derivative by finite differences
      v_omega[k] += exp(iw*(n*dt))*(v2[n+1]-2*v2[n]+v2[n-1]); 
    }
    // the third derivative, on the boundary
    v_omega[k] += (v2[1] - v2[0] + v2[N] - v2[N-1]); 
    v_omega[k] *= 1./(dt*iw*iw*iw*iw);
    // the boundary terms of the first, second, and third partial integration.
    v_omega[k] += f_omega(iw, c_(0), c_(1), c_(2)); 
    //std::cout<<"derivative at boundary: "<<-v2[1] + v2[0] + v2[N] - v2[N-1]<<" divisor: "<< 1./(dt*iw*iw*iw*iw)<<std::endl;
    // the proper convention for the self consistency loop here.
    gomega(k)=v_omega[k]; 
  }
} 

