#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <complex>


PyObject* IFT_mat2tau(PyObject *f_mat_in, const int &Nmax, const double &beta, const double &asymp_coefs1, const double &asymp_coefs2)
{
    PyArrayObject *f_mat = (PyArrayObject *) f_mat_in;
    import_array1(NULL);
    uint nfreq = f_mat->dimensions[0];
    npy_intp Nm = (npy_intp)Nmax;
    PyArrayObject *f_tau = (PyArrayObject *)PyArray_SimpleNew(1, &Nm, NPY_DOUBLE);
    Py_INCREF(f_tau);
    double *buf = (double *)PyArray_DATA(f_tau);
    std::complex<double> *f_matsubara = (std::complex<double> *)PyArray_DATA(f_mat);
    double C[2] = {asymp_coefs1, asymp_coefs2};

    for (int i = 0; i < Nmax; i++) {
        double tau=double(i)*beta/double(Nmax - 1);
        buf[i] = -C[0]/2. + C[1]/4.*(-beta+2*tau);
        for (uint k = 0; k < nfreq; ++k) {
            double wt((2*k+1)*double(i)*M_PI/double(Nmax - 1));
            std::complex<double> iomegan(0,(2*k+1)*M_PI/beta);
            std::complex<double> f_matsubara_nomodel = f_matsubara[k] - C[0]/iomegan - C[1]/(iomegan*iomegan);
            buf[i] += 2./beta*(cos(wt)*f_matsubara_nomodel.real() + sin(wt)*f_matsubara_nomodel.imag());
        }
    }
    return PyArray_Return(f_tau);
}

PyObject* FT_tau2mat(PyObject *f_tau_py_in, const double &beta, const int &Nmax)
{
    PyArrayObject *f_tau_py = (PyArrayObject *) f_tau_py_in;
    import_array1(NULL);
    double *f_tau = (double *)PyArray_DATA(f_tau_py);
    int nfreq = f_tau_py->dimensions[0] - 1;
    PyArrayObject *f_mat;
    npy_intp Nm = (npy_intp)Nmax;
    f_mat = (PyArrayObject *)PyArray_SimpleNew(1, &Nm, NPY_CDOUBLE);
    Py_INCREF(f_mat);
    std::complex<double> *f_matsubara = (std::complex<double> *)PyArray_DATA(f_mat);
    
    // trapezoidal rule for integration
    for (int n = 0; n < Nmax; n++) {
        double wn = (2*n + 1)*M_PI/beta;
        f_matsubara[n] = 0;
        for (int p = 1; p < nfreq; p++) {
            double tau = beta/double(nfreq)*p;
            f_matsubara[n] += f_tau[p]*(cos(wn*tau) + std::complex<double>(0,1)*sin(wn*tau));
        }
        f_matsubara[n] = beta/2./double(nfreq)*(f_tau[0] - f_tau[nfreq]) + beta/double(nfreq)*f_matsubara[n];
    }   
    return PyArray_Return(f_mat);
}

