#ifndef INTEGRATE_RH_GREEN_INTEGRAND_H_
#define INTEGRATE_RH_GREEN_INTEGRAND_H_

#include <eigen_integrate.h>
#include <inverse.h>
#include <iostream>

#include <Eigen/StdVector>

const int kNLayers = 4;
const int kNCor  = 3;
const int kMSize = kNLayers*kNCor;
const int kDim   = 3;

typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex, kMSize, kMSize> ComplexMatrix;
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(ComplexMatrix)


class GreenIntegrand: public md_int::GeneralIntegrand<Complex> {
 public:
  GreenIntegrand(const Eigen::VectorXcd &omega, const double &chemical_potential, 
      const std::vector<ComplexMatrix> &hamiltonian_r, const Eigen::MatrixXi &r_vectors, 
      const Eigen::MatrixXcd &all_self_energy, const double &magnetic_field, const double &double_counting);
  void CalculateOriginalIntegrand(const Eigen::VectorXd &x, Eigen::VectorXcd &y) const;
  void set_data(const int &i);

 private:
  const std::vector<ComplexMatrix> &ham_r_;
  const Eigen::MatrixXi &r_vectors_;
  const Eigen::MatrixXcd &all_self_energy_;
  const Eigen::VectorXcd &all_omega_;
  ComplexMatrix self_energy_;
  double chemical_potential_, h_field_, delta_;
  Complex w_;
};


class HamiltonianIntegrand: public md_int::GeneralIntegrand<Complex> {
 public:
  HamiltonianIntegrand(const int &num_order, const std::vector<ComplexMatrix> &hamiltonian_r, 
      const Eigen::MatrixXi &r_vectors, const double &magnetic_field, const double &double_counting);
  void CalculateOriginalIntegrand(const Eigen::VectorXd &x, Eigen::VectorXcd &y) const;

 private:
  const std::vector<ComplexMatrix> &ham_r_;
  const Eigen::MatrixXi &r_vectors_;
  double h_field_, delta_;
  int num_order_;
};


#endif  // INTEGRATE_RH_GREEN_INTEGRAND_H_
