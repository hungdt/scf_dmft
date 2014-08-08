#include <boost/python.hpp>
#include "integrand.h"

using namespace Eigen;


void GetBandHamiltonian(const VectorXd &x, ComplexMatrix &mat, const std::vector<ComplexMatrix> &HR, 
    const MatrixXi &R, const double &H, const double &delta) 
{
  static Complex I(0,1);
  mat.fill(0);
  for (int i = 0; i < R.rows(); ++i) {
    mat += HR[i]*exp(I*(x(0)*R(i,0) + x(1)*R(i,1) + x(2)*R(i,2)));
  }
  for (int i = 0; i < mat.rows(); ++i)
    mat(i,i) -= H;
  for (int i = 0; i < kNCor*kNLayers; ++i)
    mat(i,i) -= delta;
}


GreenIntegrand::GreenIntegrand(const Eigen::VectorXcd &omega, const double &chemical_potential,   
    const std::vector<ComplexMatrix> &hamiltonian_r, const Eigen::MatrixXi &r_vectors,   
    const Eigen::MatrixXcd &all_self_energy, const double &magnetic_field, const double &double_counting) 
    : md_int::GeneralIntegrand<Complex>(), delta_(double_counting), chemical_potential_(chemical_potential),
    ham_r_(hamiltonian_r), r_vectors_(r_vectors), h_field_(magnetic_field), all_omega_(omega), 
    all_self_energy_(all_self_energy), self_energy_(ComplexMatrix::Zero()) {
  set_dim_in(kDim);
  set_dim_out(kMSize*kMSize);
}

// set omega and self energy:
// e.g. tilted 5d band case: 4 blocks for 4 cell, each block is a 5x5 matrix 
void GreenIntegrand::set_data(const int &r) {
    w_ = all_omega_(r);

    for (int c = 0; c < kNLayers; ++c) 
        for (int i = 0; i < kNCor; ++i)
            for (int j = 0; j < kNCor; ++j)
                self_energy_(kNCor*c+i,kNCor*c+j) = all_self_energy_(r, kNCor*kNCor*c+kNCor*i+j);
}

void GreenIntegrand::CalculateOriginalIntegrand(const VectorXd &x, VectorXcd &y) const
{
    y.resize(dim_out());
    ComplexMatrix tmp, negative_green;
    GetBandHamiltonian(x, tmp, ham_r_, r_vectors_, h_field_, delta_);

    for (int n = 0; n < kMSize; ++n) tmp(n, n) -= w_ + chemical_potential_;
    tmp += self_energy_;

    inv::inverse(tmp, negative_green);
    for (int i = 0; i < kMSize; ++i)
        for (int j = 0; j < kMSize; ++j)
            y(kMSize*i+j) = -negative_green(i, j);
}


HamiltonianIntegrand::HamiltonianIntegrand(const int &num_order, const std::vector<ComplexMatrix> &hamiltonian_r, 
    const Eigen::MatrixXi &r_vectors, const double &magnetic_field, const double &double_counting)
    : md_int::GeneralIntegrand<Complex>(), num_order_(num_order), ham_r_(hamiltonian_r), 
    r_vectors_(r_vectors), h_field_(magnetic_field), delta_(double_counting) {
  set_dim_in(kDim);
  set_dim_out(num_order_*kMSize*kMSize);
}

void HamiltonianIntegrand::CalculateOriginalIntegrand(const VectorXd &x, VectorXcd &y) const
{
    ComplexMatrix ham, ham_pow(ComplexMatrix::Identity());
    y.resize(dim_out());
    GetBandHamiltonian(x, ham, ham_r_, r_vectors_, h_field_, delta_);
    for (int n = 0; n < num_order_; ++n) {
        ham_pow *= ham;
        for (int i = 0; i < kMSize; ++i)
            for (int j = 0; j < kMSize; ++j)
                y(kMSize*kMSize*n+kMSize*i+j) = ham_pow(i, j);
    }
}

