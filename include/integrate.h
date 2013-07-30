/*
 * n-dim Gaussian quadrature for general functions with vector output
 * using OpenMP for parallelization
 *
 * Hung Dang (May 12, 2011)
 */

#ifndef __INTEGRATE_H__
#define __INTEGRATE_H__

#include <cstring>

namespace md_int {


// 
// N : dim of space for integrating
// M : size of the integrand vector
// L : size of vector bp
// ret must be allocated to be a vector of size M
// 
inline void calculate_id(const int &n, const int &L, const int &d, int *&new_num)
{
    int div, rem;

    div = n;

    int id = 0;
    while (div != 0) {
        rem = div % L;
        div = div / L;
        new_num[id] = rem;
        id++;
    }

    for (int  i = id; i < d; ++i)
        new_num[i] = 0;
}

template<typename T>
T* integrate(const int N, const int M, 
        const double *xl, const double *xh, 
        const int &L, const double *bp, const double *wf,
        void (*integrand)(const int &, const double *in, const int &, T *out, const void *data),
        void *integrand_data)
{
    double *qx = new double[N],
        *px = new double[N];
    for (int i = 0; i < N; ++i) {
        qx[i] = (xh[i] - xl[i])/2.;
        px[i] = (xh[i] + xl[i])/2.;
    }
    T *ret = new T[M];
    memset(ret, 0, M*sizeof(T));

#pragma omp parallel default(shared)
    {
        double *x = new double[N];
        double weight;
        int *id = new int[N];
        T *tmp = new T[M];
        T *ret_local = new T[M];
        memset(ret_local, 0, M*sizeof(T));

#pragma omp for
        for (int n = 0; n < int(pow(L, N)); ++n) {
            // calculate the index id for each n
            calculate_id(n, L, N, id);
            // calculate the integrand and add to vector ret
            weight = 1.;
            for (int i = 0; i < N; ++i) {
                x[i] = qx[i]*bp[id[i]] + px[i];
                weight *= wf[id[i]];
            }
            integrand(N, x, M, tmp, integrand_data);
            for (int i = 0; i < M; ++i)
                ret_local[i] += tmp[i]*weight;
        }
#pragma omp critical 
        {
            for (int i = 0; i < M; ++i)
                ret[i] += ret_local[i];
        }
    
        delete[] x;
        delete[] id;
        delete[] tmp;
        delete[] ret_local;
    }
    
    // Jacobian
    for (int i = 0; i < M; ++i) 
        for (int j = 0; j < N; ++j)
            ret[i] *= qx[j];

    
    delete[] qx;
    delete[] px;

    return ret;
}

} // end of namespace md_int

#endif
