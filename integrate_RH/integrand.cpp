#include <boost/python.hpp>
#include "integrand.h"

using namespace Eigen;


void getHband(const VectorXd &x, MyMatcd &mat, const std::vector<MyMatcd> &HR, const MatrixXi &R, const double &H, const double &delta) 
{
    mat.fill(0);
    for (int i = 0; i < R.rows(); ++i) {
        mat += HR[i]*exp(complex_t(0.,1.)*(x(0)*R(i,0) + x(1)*R(i,1) + x(2)*R(i,2)));
    }
    for (int i = 0; i < mat.rows(); ++i)
        mat(i,i) -= H;
    for (int i = 0; i < NCOR*N_LAYERS; ++i)
        mat(i,i) -= delta;
}


GreenIntegrand::GreenIntegrand(const VectorXcd &w_, const double &mu_, const std::vector<MyMatcd> &HR_, const MatrixXi &R_, const MatrixXcd &SE_, const double &Hf, const double &delta_)
    : md_int::integrand<complex_t>(), delta(delta_), mu(mu_), HR(HR_), R(R_), H(Hf), wall(w_), SEall(SE_), SE(MyMatcd::Zero())
{
    dim_in = DIM;
    dim_out = MSIZE*MSIZE;
}


void GreenIntegrand::set_data(const int &r)  // 4 blocks for 4 cell, each block is a 5x5 matrix (5 d-bands)
{
    w = wall(r);

    for (int c = 0; c < N_LAYERS; ++c) 
        for (int i = 0; i < NCOR; ++i)
            for (int j = 0; j < NCOR; ++j)
                SE(NCOR*c+i,NCOR*c+j) = SEall(r, NCOR*NCOR*c+NCOR*i+j);
}


void GreenIntegrand::calculate_integrand(const VectorXd &x, VectorXcd &y) const
{
    y.resize(dim_out);
    MyMatcd tmp, negG;
    getHband(x, tmp, HR, R, H, delta);

    for (int L = 0; L < MSIZE; ++L) tmp(L, L) -= w + mu;
    tmp += SE;

    inv::inverse(tmp, negG);
    for (int i = 0; i < MSIZE; ++i)
        for (int j = 0; j < MSIZE; ++j)
            y(MSIZE*i+j) = -negG(i, j);
}



HIntegrand::HIntegrand(const int &Norder, const std::vector<MyMatcd> &HR_, const MatrixXi &R_, const double &Hf_in, const double &delta_)
    : md_int::integrand<complex_t>(), N_order(Norder), HR(HR_), R(R_), Hf(Hf_in), delta(delta_)
{
    dim_in = DIM;
    dim_out = N_order*MSIZE*MSIZE;
}


void HIntegrand::calculate_integrand(const VectorXd &x, VectorXcd &y) const
{
    MyMatcd H;
    MyMatcd Hpow(MyMatcd::Identity());

    y.resize(dim_out);
    getHband(x, H, HR, R, Hf, delta);
    for (int n = 0; n < N_order; ++n) {
        Hpow *= H;
        for (int i = 0; i < MSIZE; ++i)
            for (int j = 0; j < MSIZE; ++j)
                y(MSIZE*MSIZE*n+MSIZE*i+j) = Hpow(i, j);
    }
}

