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

#ifndef FOURIER_TRANSFORM_H
#define FOURIER_TRANSFORM_H
#include <complex>
#include <Eigen/Core>
#include <iostream>


inline std::complex<double> f_omega(std::complex<double> iw, 
                                    double c1, double c2, double c3) {
  std::complex<double> iwsq=iw*iw;
  return c1/iw + c2/(iwsq) + c3/(iw*iwsq);
}

inline double f_tau(double tau, double beta, 
                    double c1, double c2, double c3) {
  return -0.5*c1 + (c2*0.25)*(-beta+2.*tau) + (c3*0.25)*(beta*tau-tau*tau);
}


class FourierTransformer {
 public:
  FourierTransformer(const double beta, const Eigen::VectorXd &ftail);
  virtual ~FourierTransformer() {}
  void forward_ft(const Eigen::VectorXd &G_tau, Eigen::VectorXcd &G_omega,
                  const int num_matsubara) const;
  void Armin_forward_ft(const Eigen::VectorXd &G_tau, Eigen::VectorXcd &G_omega,
                  const int num_matsubara) const;
  void backward_ft(const Eigen::VectorXcd &G_omega, 
                   Eigen::VectorXd &G_tau, const int N_tau) const;
 protected:
  double beta_;
  // coefficients for the Green's functions
  const Eigen::VectorXd &c_;
};

#endif
