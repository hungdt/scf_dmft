#ifndef __GREEN_INTEGRAND_H__
#define __GREEN_INTEGRAND_H__

#include <eigen_integrate.h>
#include <inverse.h>
#include <iostream>

#include <Eigen/StdVector>

const int MSIZE = 3;
const int DIM   = 2;

typedef std::complex<double> complex_t;

class GreenIntegrand: public md_int::GeneralIntegrand<complex_t> {
    public:
        GreenIntegrand(const Eigen::VectorXcd &w_, const double &mu_, 
            const Eigen::MatrixXcd &SE, 
            const Eigen::VectorXd &SK, const double &Hf);
        void CalculateOriginalIntegrand(const Eigen::VectorXd &x, 
                                        Eigen::VectorXcd &y) const;
        void set_data(const int &i);
    private:
        Eigen::MatrixXcd SEall;
        Eigen::VectorXcd wall;
        Eigen::Matrix<complex_t, MSIZE, 1> SE;
        Eigen::VectorXd SK;
        double mu, H;
        complex_t w;
};


class HIntegrand: public md_int::GeneralIntegrand<complex_t> {
    public:
        HIntegrand(const int &Norder, const Eigen::VectorXd &SK_, 
                   const double &H_in);
        void CalculateOriginalIntegrand(const Eigen::VectorXd &x, 
                                        Eigen::VectorXcd &y) const;
    private:
        Eigen::VectorXd SK;
        double Hf;
        int N_order;
};


#endif
