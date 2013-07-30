#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <numpy_eigen.h>
#include <omp.h>

#include "integrand.h"

using namespace Eigen;


PyObject* calc_Gavg(PyObject *pyw, const double &delta, const double &mu, PyObject *pySE, PyObject *pyTB, const double &Hf, PyObject *py_bp, PyObject *py_wf, const int nthreads)
{
    try {
        if (nthreads > 0) omp_set_num_threads(nthreads);
        VectorXd xl(DIM), xh(DIM);
        xl << 0, 0;
        xh << M_PI, M_PI;
        double BZ1 = (xh - xl).prod();
        MatrixXcd SE;
        VectorXcd w;
        VectorXd bp, wf;
        VectorXd SlaterKosterCoeffs;

        numpy::from_numpy(pySE, SE);
        numpy::from_numpy(pyTB, SlaterKosterCoeffs);
        numpy::from_numpy(py_bp, bp);
        numpy::from_numpy(py_wf, wf);
        numpy::from_numpy(pyw, w);
        assert(w.size() == SE.rows());

        MatrixXcd result(SE.rows(), MSIZE*MSIZE);
        VectorXcd tmp(MSIZE);
        GreenIntegrand green_integrand(w, mu, SE, SlaterKosterCoeffs, Hf);
        result.setZero();
        for (int n = 0; n < w.size(); ++n) {
            green_integrand.set_data(n);
            tmp = md_int::integrate(xl, xh, green_integrand, bp, wf);
            for (int i = 0; i < MSIZE; ++i)
                result(n, MSIZE*i + i) = tmp(i);
        }
        result /= BZ1;
        return numpy::to_numpy(result);
    } catch (const char *str) {
        std::cerr << str << std::endl;
        return Py_None;
    }
}

PyObject* calc_Havg(const int &Norder, PyObject *pyTB, const double &Hf, const double &delta, PyObject *py_bp, PyObject *py_wf)
{
    try {
        VectorXd xl(DIM), xh(DIM);
        xl << 0, 0;
        xh << M_PI, M_PI;
        double BZ1 = (xh - xl).prod();
        MatrixXi R;
        VectorXd bp, wf;
        VectorXd SlaterKosterCoeffs;

        numpy::from_numpy(pyTB, SlaterKosterCoeffs);
        numpy::from_numpy(py_bp, bp);
        numpy::from_numpy(py_wf, wf);

        HIntegrand h_integrand(Norder, SlaterKosterCoeffs, Hf);
        VectorXd result = (md_int::integrate(xl, xh, h_integrand, bp, wf)).real();
        result /= BZ1;
        VectorXd ret(Norder*MSIZE*MSIZE);
        ret.setZero();
        for (int i = 0; i < Norder; ++i)
            for (int j = 0; j < MSIZE; ++j)
                ret(MSIZE*MSIZE*i + MSIZE*j + j) = result(MSIZE*i + j);
        return numpy::to_numpy(ret);
    } catch (const char *str) {
        std::cerr << str << std::endl;
        return Py_None;
    }
}

BOOST_PYTHON_MODULE(int_2dhopping)
{
    using namespace boost::python;
    def("calc_Gavg", calc_Gavg);
    def("calc_Havg", calc_Havg);
}
