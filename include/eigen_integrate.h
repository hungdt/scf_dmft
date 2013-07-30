/*
 * Eigen wrapper for integrate
 * need eigen3 (see http://eigen.tuxfamily.org)
 *
 * Hung Dang (May 12, 2011)
 */

#ifndef __EIGEN_INTEGRATE_H__
#define __EIGEN_INTEGRATE_H__

#include <Eigen/Core>
#include <integrate.h>

namespace md_int {

template<typename T>
class integrand {
    public:
        integrand()
            : dim_in(0), dim_out(0)
        {}

        integrand(const int &d_in, const int &d_out) 
            : dim_in(d_in), dim_out(d_out) 
        {}

        int get_dim_out() const { return dim_out; }
        int get_dim_in() const { return dim_in; }
        void set_dim_in(const int &d_in) { dim_in = d_in; }
        void set_dim_out(const int &d_out) { dim_out = d_out; }

        T* operator()(const double *x) const {
            T* y = new T[dim_out];
            calculate_integrand(x, y);
            return y;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1>* operator()(const Eigen::VectorXd &x) const {
            Eigen::Matrix<T, Eigen::Dynamic, 1>* y = new Eigen::Matrix<T, Eigen::Dynamic, 1>(this->dim_out);
            calculate_integrand(x, *y);
            return y;
        }

        virtual void calculate_integrand(const Eigen::VectorXd &x, Eigen::Matrix<T, Eigen::Dynamic, 1> &y) const {}

        virtual void calculate_integrand(const double *x, T* y) const {
            Eigen::Map<Eigen::VectorXd > xx((double *)x, dim_in);
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> > yy(y, dim_out);
            calculate_integrand((Eigen::VectorXd &)xx, (Eigen::Matrix<T, Eigen::Dynamic, 1> &)yy);
        }

    protected:
        int dim_in, dim_out;
};


template<typename T>
class block_integrand : public integrand<T> {
    public:
        block_integrand()
            : num_block(0)
        {}

        block_integrand(const int &nb)
            : num_block(nb), block_dim_in(nb), block_dim_out(nb), current_block(0)
        {}

        int get_num_block() const {
            return num_block;
        }

        int get_block() const {
            return current_block;
        }

        int get_max_dim_in() const {
            return block_dim_in.maxCoeff();
        }

        int get_total_dim_out() const {
            return block_dim_out.sum();
        }

        void set_block(const int &i) {
            current_block = i;
            this->dim_in = block_dim_in[i];
            this->dim_out = block_dim_out[i];
        }

        void set_num_block(const int &i) {
            num_block = i;
            block_dim_in.resize(i);
            block_dim_out.resize(i);
            current_block = 0;
        }


    protected:
        int num_block;
        Eigen::VectorXi block_dim_in, block_dim_out;

    private:
        int current_block;
};


template<typename T> 
void integrand_wrapper(const int &N, const double *in, const int &M, T *out, const void *data)
{
    integrand<T> *my_integrand = (integrand<T> *)data;
    if (M != my_integrand->get_dim_out()) 
        throw "Output size not matched.";
    if (N != my_integrand->get_dim_in())
        throw "Input size not matched.";

    my_integrand->calculate_integrand(in, out);
}

template<typename T> 
Eigen::Matrix<T, Eigen::Dynamic, 1> integrate(const Eigen::VectorXd &xl, const Eigen::VectorXd &xh, const integrand<T> &my_integrand, 
        const Eigen::VectorXd &bp, const Eigen::VectorXd &wf)
{
    int N = my_integrand.get_dim_in(),
        M = my_integrand.get_dim_out(),
        L = bp.size();
    if (N > xl.size() || xl.size() != xh.size())
        throw "Variable dim not consistent";
    
    T *ret_ptr = integrate(N, M, xl.data(), xh.data(), L, bp.data(), wf.data(), &integrand_wrapper<T>, (void *)&my_integrand);
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> >(ret_ptr, M);
}

template<typename T> 
Eigen::Matrix<T, Eigen::Dynamic, 1> block_integrate(const Eigen::VectorXd &xl, const Eigen::VectorXd &xh, 
        block_integrand<T> &my_integrand, 
        const Eigen::VectorXd &bp, const Eigen::VectorXd &wf)
{
    if (xl.size() != xh.size())
        throw "Size of lower-bound vector not equal to size of upper-bound vector";
    if (xl.size() < my_integrand.get_max_dim_in())
        throw "Not enough values for integrating interval";

    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(my_integrand.get_total_dim_out());

    Eigen::VectorXd int_region(xh-xl);
    double volume = int_region.prod();
    int pos = 0;
    for (int i = 0; i < my_integrand.get_num_block(); ++i) {
        my_integrand.set_block(i);
        Eigen::Matrix<T, Eigen::Dynamic, 1>  tmp(integrate<T>(xl, xh, my_integrand, bp, wf));
        double indepvar_contrib = volume / int_region.head(my_integrand.get_dim_in()).prod();
        for (int j = 0; j < my_integrand.get_dim_out(); ++j)
            ret[pos+j] = indepvar_contrib*tmp[j];
        pos += my_integrand.get_dim_out();
    }

    return ret;
}

} // end of namespace md_int

#endif
