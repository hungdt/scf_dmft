#include <boost/python.hpp>
#include "integrand.h"

using namespace Eigen;

void getHband(const VectorXd &x, complex_t &mat, const VectorXd &SK, const double &H)
{
    // SK : t, t1
    double t = SK[0], t1 = SK[1];
    mat = -2*t*(cos(x(0)) + cos(x(1))) - 4*t1*cos(x(0))*cos(x(1)) - H;
}


GreenIntegrand::GreenIntegrand(const VectorXcd &w_, const double &mu_, const MatrixXcd &SE_, const VectorXd &SK_, const double &Hf)
    : md_int::integrand<complex_t>(), mu(mu_), H(Hf), wall(w_), SEall(SE_), SK(SK_)
{
    dim_in  = DIM;
    dim_out = MSIZE;
}


void GreenIntegrand::set_data(const int &r)
{
    w = wall(r);
    for (int i = 0; i < MSIZE; ++i)
        SE(i) = SEall(r, MSIZE*i+i);  // diagonal terms
}


void GreenIntegrand::calculate_integrand(const VectorXd &x, VectorXcd &y) const
{
    y.resize(dim_out);
    complex_t Hband;
    getHband(x, Hband, SK, H);

    for (int L = 0; L < MSIZE; ++L) { 
        y(L) = 1. / (w + mu - Hband - SE(L));
    }
}


HIntegrand::HIntegrand(const int &Norder, const VectorXd &SK_, const double &Hf_in)
    : md_int::integrand<complex_t>(), N_order(Norder), Hf(Hf_in), SK(SK_)
{
    dim_in  = DIM;
    dim_out = N_order*MSIZE;
}


void HIntegrand::calculate_integrand(const VectorXd &x, VectorXcd &y) const
{
    complex_t Hband, Hpow = 1.;
    y.resize(dim_out);
    getHband(x, Hband, SK, Hf);
    for (int n = 0; n < N_order; ++n) {
        Hpow *= Hband;
        for (int i = 0; i < MSIZE; ++i)
            y(MSIZE*n+i) = Hpow;
    }
}

