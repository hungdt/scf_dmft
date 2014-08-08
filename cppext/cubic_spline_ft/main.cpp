#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <numpy_eigen.h>
#include "fouriertransform.h"

PyObject* ArminForwardFourierTransform(PyObject *py_ftau, const double beta, 
                                  PyObject *py_ftail, const int num_matsubara) {
  Eigen::VectorXd ftau, ftail;
  numpy::from_numpy(py_ftau, ftau);
  numpy::from_numpy(py_ftail, ftail);

  Eigen::VectorXcd fmat;
  FourierTransformer fourier(beta, ftail);
  fourier.Armin_forward_ft(ftau, fmat, num_matsubara);
  
  return numpy::to_numpy(fmat);
}

PyObject* ForwardFourierTransform(PyObject *py_ftau, const double beta, 
                                  PyObject *py_ftail, const int num_matsubara) {
  Eigen::VectorXd ftau, ftail;
  numpy::from_numpy(py_ftau, ftau);
  numpy::from_numpy(py_ftail, ftail);

  Eigen::VectorXcd fmat;
  FourierTransformer fourier(beta, ftail);
  fourier.forward_ft(ftau, fmat, num_matsubara);
  
  return numpy::to_numpy(fmat);
}

PyObject* BackwardFourierTransform(PyObject *py_fmat, const double beta, 
                                   PyObject *py_ftail, const int num_tau) {
  Eigen::VectorXcd fmat;
  Eigen::VectorXd ftail;
  numpy::from_numpy(py_fmat, fmat);
  numpy::from_numpy(py_ftail, ftail);

  Eigen::VectorXd ftau;
  FourierTransformer fourier(beta, ftail);
  fourier.backward_ft(fmat, ftau, num_tau);
  return numpy::to_numpy(ftau);
}

BOOST_PYTHON_MODULE(fourier)
{
  boost::python::def("Armin_forward_ft", ArminForwardFourierTransform);
  boost::python::def("forward_ft", ForwardFourierTransform);
  boost::python::def("backward_ft", BackwardFourierTransform);
}
