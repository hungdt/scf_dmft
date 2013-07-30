#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <cassert>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numpy_eigen.h>
#include <vector>
#include <iostream>

PyObject* ewald_sum(const int &NLa, const double &CLa, const int &NSr, const double &CSr, PyObject *nV, const int &Nxy, const int &Nz)
{
    assert(((PyArrayObject *)nV)->dimensions[0] == NLa + NSr);
    import_array1(NULL);
    double sigma = M_PI;
    int N = NLa + NSr;
    double *C = new double[2*N],
           *pt_nV = (double *) PyArray_DATA((PyArrayObject *) nV);
    double *x = new double[2*N],
           *y = new double[2*N],
           *z = new double[2*N],
           *Vsi = new double[N];
    std::complex<double>  *Vli = new std::complex<double>[N];
    double sum_nV = 0.;
    for (int i = 0; i < N; ++i)
        sum_nV += pt_nV[i];

    for (int i = 0; i < N; ++i) {
        C[i] = pt_nV[i]/sum_nV * (-NLa*CLa - NSr*CSr);
        C[N+i] = (i < NLa) ? CLa : CSr;
        x[i] = y[i] = 0.;
        x[N+i] = y[N+i] = 0.5;
        z[i] = double(i);
        z[N+i] = -0.5+double(i);
        Vsi[i] = 0.;
        Vli[i] = std::complex<double>(0., 0.);
    }


    double e_charge = -1.;
    int Nx = Nxy, Ny = Nxy;
    double *R = new double[2*N],
           kv[3], ksquare = 0.;

    for (int i = 0; i < N; ++i) 
        for (int m = -Nx; m < Nx; ++m) 
            for (int n = -Ny; n < Ny; ++n) 
                for (int l = - Nz; l < Nz; ++l) {
                    for (int k = 0; k < 2*N; ++k)
                        R[k] = sqrt(pow(m+x[k], 2) + pow(n+y[k],2) + pow(N*l+z[i]-z[k], 2));
                    // for short-range terms
                    if (m == 0 and n == 0 and l == 0) {
                        for (int k = 0; k < 2*N; ++k)
                            if (k != i)
                                Vsi[i] += e_charge*C[k] / R[k] * erfc(R[k]/sqrt(2)/sigma);
                        continue;
                    }
                    for (int k = 0; k < 2*N; ++k)
                        Vsi[i] += e_charge * C[k] / R[k] * erfc(R[k]/sqrt(2)/sigma);
                    
                    // for long-range terms
                    kv[0] = 2*M_PI*n;
                    kv[1] = 2*M_PI*m;
                    kv[2] = 2*M_PI/N*l;
                    ksquare = pow(kv[0], 2) + pow(kv[1], 2) + pow(kv[2], 2);
                    for (int k = 0; k < 2*N; ++k) 
                        Vli[i] += 4*M_PI/N * e_charge*C[k] / ksquare * 
                            exp(std::complex<double>(0., -kv[0]*x[k] - kv[1]*y[k] + kv[2]*(z[i] - z[k]))) * exp(-pow(sigma,2)*ksquare/2.);
                }
    npy_intp Nconvert = npy_intp(N);
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, &Nconvert, NPY_DOUBLE);
    Py_INCREF(result);
    double *result_buf = (double *)PyArray_DATA(result);
    for (int i = 0; i < N; ++i)
        result_buf[i] = Vsi[i] + Vli[i].real() - e_charge*C[i]*sqrt(2/M_PI)*1./sigma;

    delete[] R;
    delete[] C;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] Vsi;
    delete[] Vli;

    return PyArray_Return(result);
}


using namespace Eigen;

typedef struct {
    Vector3d r; // position in the unitcell
    double C;   // ion charge
    double ne;  // electron occupation 
    double V;   // dimensionless electric potential (whose unit is smt like V :)
} AtomInfo;
typedef std::vector<AtomInfo> UnitCell;

void general_ewald_sum(UnitCell &uc, const Vector3d &a1, const Vector3d &a2, const Vector3d &a3, const Vector3i &Rcutoff);


PyObject* new_ewald_sum(const int &NLa, const double &CLa, const int &NSr, const double &CSr, PyObject *nVpy, const int &Nxy, const int &Nz)
{
    try {
        int N = NLa + NSr;
        VectorXd nV;
        numpy::from_numpy(nVpy, nV);

        UnitCell uc(2*N);
        for (int i = 0; i < N; ++i) {
            uc[2*i].C = (i < NLa) ? CLa : CSr;
            uc[2*i].ne = 0;
            uc[2*i].r << 0.5, 0.5, -0.5 + double(i);

            uc[2*i+1].C = 0;
            uc[2*i+1].ne = nV(i);
            uc[2*i+1].r << 0, 0, i;
        }

        Vector3d a1(1,0,0), a2(0,1,0), a3(0,0,N);
        Vector3i Rcutoff(Nxy, Nxy, Nz);
        general_ewald_sum(uc, a1, a2, a3, Rcutoff);

        VectorXd out(N);
        for (int i = 0; i < N; ++i) 
            out(i) = -uc[2*i+1].V;
        return numpy::to_numpy(out);
    } catch (const char *str) {
        std::cerr << str << std::endl;
        return Py_None;
    }
}


void general_ewald_sum(UnitCell &uc, const Vector3d &a1, const Vector3d &a2, const Vector3d &a3, const Vector3i &Rcutoff)
{
    int N = uc.size();
    const double sigma = M_PI;

    // renormalize electron occupation
    double unrenorm_Ne = 0, Ne = 0;
    for (UnitCell::iterator it = uc.begin(); it < uc.end(); ++it) {
        unrenorm_Ne += it->ne;
        Ne += it->C;
    }
    for (UnitCell::iterator it = uc.begin(); it < uc.end(); ++it) 
        it->ne *= Ne/unrenorm_Ne;

    // get reciprocal vectors
    Vector3d b1, b2, b3;
    double uc_vol = a1.dot(a2.cross(a3));
    b1 = 2*M_PI*a2.cross(a3)/uc_vol;
    b2 = 2*M_PI*a3.cross(a1)/uc_vol;
    b3 = 2*M_PI*a1.cross(a2)/uc_vol;
    
    // Ewald sum
    double Vs, Vl;
    Vector3d k;
    for (int id = 0; id < N; ++id) {
        Vs = 0; Vl = 0;
        for (int m = -Rcutoff[0]; m < Rcutoff[0]; ++m)
            for (int n = -Rcutoff[1]; n < Rcutoff[1]; ++n)
                for (int l = -Rcutoff[2]; l < Rcutoff[2]; ++l) {
                    k = n*b1 + m*b2 + l*b3;
                    double ksquare = k.dot(k);
                    for (int i = 0; i < N; ++i) {
                        double dR = (uc[id].r - (uc[i].r + m*a1 + n*a2 + l*a3)).norm();
                        // for long-range terms
                        if ( !(m == 0 && n == 0 && l == 0) )
                            Vl += 4*M_PI/uc_vol*(uc[i].C-uc[i].ne)/ksquare * cos(k.dot(uc[id].r-uc[i].r)) * exp(-pow(sigma,2)*ksquare/2.);
                        
                        // for short-range terms
                        if ( !(m == 0 && n == 0 && l == 0 && i == id) )
                            Vs += (uc[i].C - uc[i].ne) / dR * erfc(dR/sqrt(2)/sigma);
                    }
                }
        uc[id].V = Vs + Vl - (uc[id].C - uc[id].ne)*sqrt(2/M_PI)/sigma;
    }
}

