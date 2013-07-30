#include <boost/python.hpp>
#include <numpy/arrayobject.h>

using namespace boost::python;


PyObject* ewald_sum(const int &NLa, const double &CLa, const int &NSr, const double &CSr, PyObject *nV, const int &Nxy, const int &Nz);
PyObject* new_ewald_sum(const int &NLa, const double &CLa, const int &NSr, const double &CSr, PyObject *nVpy, const int &Nxy, const int &Nz);
PyObject* IFT_mat2tau(PyObject *f_mat_in, const int &Nmax, const double &beta, const double &asymp_coefs1, const double &asymp_coefs2);
PyObject* FT_tau2mat(PyObject *f_tau_py_in, const double &beta, const int &Nmax);
PyObject * get_raw_data(boost::python::str &path, const int &measure, PyObject *obs_str_list);
double norm(PyObject *v);
double interp_root(PyObject *x, PyObject *y, const double &y0);
PyObject* array_erfc(PyObject *x);


BOOST_PYTHON_MODULE(cppext)
{
    using namespace boost::python;
    def("IFT_mat2tau", IFT_mat2tau);
    def("FT_tau2mat", FT_tau2mat);
    def("get_raw_data", get_raw_data);
    def("ewald_sum", ewald_sum);
    def("new_ewald_sum", new_ewald_sum);
}
