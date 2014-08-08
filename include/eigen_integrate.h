/*
 * Eigen wrapper for integrate
 * need eigen3 (see http://eigen.tuxfamily.org)
 *
 * Hung Dang (Oct 29, 2013)
 */

#ifndef __EIGEN_INTEGRATE_H__
#define __EIGEN_INTEGRATE_H__

#include <Eigen/Core>
#include <integrate.h>

namespace md_int {

template<typename ReturnType> class GeneralIntegrand {
 public:
  typedef Eigen::Matrix<ReturnType, Eigen::Dynamic, 1> ReturnVectorType;
  GeneralIntegrand() : dim_in_(0), dim_out_(0) {}

  int dim_out() const { return dim_out_; }
  int dim_in() const { return dim_in_; }

  void CalculateTransformedIntegrand(const Eigen::VectorXd &x, ReturnVectorType &y) const {
    Eigen::VectorXd x_original = this->XTransformation(x);
    ReturnType jacobian = this->GetJacobian(x);
    this->CalculateOriginalIntegrand(x_original, y);
    y *= jacobian;
  }

  void WrapperTransformedIntegrand(const double *x, ReturnType* y) const {
    Eigen::Map<Eigen::VectorXd > xx((double *)x, dim_in_);
    Eigen::Map<ReturnVectorType> yy(y, dim_out_);
    CalculateTransformedIntegrand((Eigen::VectorXd &)xx, (Eigen::Matrix<ReturnType, Eigen::Dynamic, 1> &)yy);
  }

  virtual void CalculateOriginalIntegrand(const Eigen::VectorXd &x, ReturnVectorType &y) const = 0;
  ReturnType* operator()(const double *x) const {
    ReturnType* y = new ReturnType[dim_out];
    WrapperTransformedIntegrand(x, y);
    return y;
  }

  ReturnVectorType operator()(const Eigen::VectorXd &x) const {
    ReturnVectorType y;
    CalculateTransformedIntegrand(x, y);
    return y;
  }

 protected:
  void set_dim_in(const int d_in) { dim_in_ = d_in; }
  void set_dim_out(const int d_out) { dim_out_ = d_out; }
  virtual ReturnType GetJacobian(const Eigen::VectorXd &x) const { return 1.; }
  virtual Eigen::VectorXd XTransformation(const Eigen::VectorXd &x) const { return x; }

 private:
  int dim_in_, dim_out_; 
};


template<typename ReturnType> void IntegrandWrapper(const int &n_in, const double *in, 
    const int &n_out, ReturnType *out, const void *data) {
  GeneralIntegrand<ReturnType> *my_integrand = (GeneralIntegrand<ReturnType> *)data;
  if (n_out != my_integrand->dim_out()) 
    throw "Output size not matched.";
  if (n_in != my_integrand->dim_in())
    throw "Input size not matched.";
  my_integrand->WrapperTransformedIntegrand(in, out);
}

template<typename ReturnType> Eigen::Matrix<ReturnType, Eigen::Dynamic, 1> Integrate(
    const Eigen::VectorXd &xl, const Eigen::VectorXd &xh, const GeneralIntegrand<ReturnType> &my_integrand, 
    const Eigen::VectorXd &bp, const Eigen::VectorXd &wf) {
  int n_in = my_integrand.dim_in(),
      n_out = my_integrand.dim_out(),
      n_gauss = bp.size();
  if (n_in > xl.size() || xl.size() != xh.size()) 
    throw "Variable dim not consistent";
  ReturnType *ret_ptr = integrate(n_in, n_out, xl.data(), xh.data(), 
      n_gauss, bp.data(), wf.data(), &IntegrandWrapper<ReturnType>, (void *)&my_integrand);
  return Eigen::Map<Eigen::Matrix<ReturnType, Eigen::Dynamic, 1> >(ret_ptr, n_out);
}

} // namespace md_int
#endif
