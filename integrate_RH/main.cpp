#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <numpy_eigen.h>
#include <omp.h>

#include "integrand.h"

using namespace Eigen;

template<typename T, int Rows, int Cols>
void convert3dArray(PyObject *pyArr, std::vector<Matrix<T, Rows, Cols> > &out)
{
    import_array();
    if (PyArray_NDIM(pyArr)!=3) throw "dim != 3";

    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(pyArr);
    npy_intp *sz = arr->dimensions;
    npy_intp strides[3];
    int elsize = (PyArray_DESCR(pyArr))->elsize;

    for (int i = 0; i < 3; ++i) strides[i] = PyArray_STRIDE(arr, i)/(PyArray_DESCR(arr))->elsize;
    out.resize(sz[0]);
    T *source = (T *)PyArray_DATA(arr);
    for (int i=0; i<sz[0]; ++i) {
        for (int r=0; r<sz[1]; r++)
            for (int c=0; c<sz[2]; c++) {
                out[i](r, c) = source[i*strides[0] + r*strides[1] + c*strides[2]];
            }
    }
}

PyObject* calc_Gavg(PyObject *pyw, const double &delta, const double &mu, PyObject *pySE, PyObject *pyHR, PyObject *pyR, const double &Hf, PyObject *py_bp, PyObject *py_wf, const int nthreads)
{
    try {
        if (nthreads > 0) omp_set_num_threads(nthreads);
        VectorXd xl(DIM), xh(DIM);
        xl << -M_PI, -M_PI, -M_PI;
        xh << M_PI, M_PI, M_PI;
        double BZ1 = (xh - xl).prod();
        std::vector<MyMatcd> HR;
        MatrixXcd SE;
        MatrixXi R;
        VectorXcd w;
        VectorXd bp, wf;

        convert3dArray(pyHR, HR);
        numpy::from_numpy(pySE, SE);
        numpy::from_numpy(pyR, R);
        numpy::from_numpy(py_bp, bp);
        numpy::from_numpy(py_wf, wf);
        numpy::from_numpy(pyw, w);

        assert(w.size() == SE.rows());

        MatrixXcd result(SE.rows(), MSIZE*MSIZE);
        GreenIntegrand green_integrand(w, mu, HR, R, SE, Hf, delta);
        for (int i = 0; i < w.size(); ++i) {
            green_integrand.set_data(i);
            result.row(i) = md_int::integrate(xl, xh, green_integrand, bp, wf);
        }
        result /= BZ1;
        return numpy::to_numpy(result);
    } catch (const char *str) {
        std::cerr << str << std::endl;
        return Py_None;
    }
}

PyObject* calc_Havg(const int &Norder, PyObject *pyHR, PyObject *pyR, const double &Hf, const double &delta, PyObject *py_bp, PyObject *py_wf)
{
    try {
        VectorXd xl(DIM), xh(DIM);
        xl << -M_PI, -M_PI, -M_PI;
        xh << M_PI, M_PI, M_PI;
        double BZ1 = (xh - xl).prod();
        std::vector<MyMatcd> HR;
        MatrixXi R;
        VectorXd bp, wf;

        convert3dArray(pyHR, HR);
        numpy::from_numpy(pyR, R);
        numpy::from_numpy(py_bp, bp);
        numpy::from_numpy(py_wf, wf);

        HIntegrand h_integrand(Norder, HR, R, Hf, delta);
        VectorXcd result = (md_int::integrate(xl, xh, h_integrand, bp, wf));
        result /= BZ1;
        return numpy::to_numpy(result);
    } catch (const char *str) {
        std::cerr << str << std::endl;
        return Py_None;
    }
}

BOOST_PYTHON_MODULE(int_donly_cubic_3bands)
{
    using namespace boost::python;
    def("calc_Gavg", calc_Gavg);
    def("calc_Havg", calc_Havg);
}
