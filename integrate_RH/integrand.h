#ifndef __GREEN_INTEGRAND_H__
#define __GREEN_INTEGRAND_H__

#include <eigen_integrate.h>
#include <inverse.h>
#include <iostream>

#include <Eigen/StdVector>

const int N_LAYERS = 1;
const int NCOR  = 3;
const int MSIZE = N_LAYERS*NCOR;
const int DIM   = 3;

typedef std::complex<double> complex_t;
typedef Eigen::Matrix<complex_t, MSIZE, MSIZE> MyMatcd;
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(MyMatcd)

class GreenIntegrand: public md_int::integrand<complex_t> {
    public:
        GreenIntegrand(const Eigen::VectorXcd &w_, const double &mu_, const std::vector<MyMatcd> &HR_, const Eigen::MatrixXi &R_,const Eigen::MatrixXcd &SE, const double &Hf, const double &delta_);
        void calculate_integrand(const Eigen::VectorXd &x, Eigen::VectorXcd &y) const;
        void set_data(const int &i);
    private:
        std::vector<MyMatcd> HR;
        Eigen::MatrixXi R;
        Eigen::MatrixXcd SEall;
        Eigen::VectorXcd wall;
        MyMatcd SE;
        double mu, H, delta;
        complex_t w;
};


class HIntegrand: public md_int::integrand<complex_t> {
    public:
        HIntegrand(const int &Norder, const std::vector<MyMatcd> &HR_, const Eigen::MatrixXi &R_, const double &H_in, const double &delta_);
        void calculate_integrand(const Eigen::VectorXd &x, Eigen::VectorXcd &y) const;

    private:
        std::vector<MyMatcd> HR;
        Eigen::MatrixXi R;
        double Hf, delta;
        int N_order;
};


#endif
