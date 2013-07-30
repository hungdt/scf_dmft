/* 
 * several ways to inverse a matrix
 * work only for square matrix
 * no pseudoinverse here
 * Hung Dang
 * Apr 21, 2011
 */

#ifndef __INVERSE_H__
#define __INVERSE_H__

#include <Eigen/Core>
#include <cstring>

#ifdef EIGEN_UMFPACK_SUPPORT
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/UmfPackSupport>
#endif

namespace inv {

extern "C" {
    void zgetrf_(const int &M, const int &N, std::complex<double> *A, const int &LDA, int *IPIV, int &INFO);
    void zgbtrf_(const int &M, const int &N, const int &KL, const int &KU, std::complex<double> *A, const int &LDA, int *IPIV, int &INFO);
    void zgbtrs_(const char &TRANS, const int &N, const int &KL, const int &KU, const int &NRHS, 
            const std::complex<double> *A, const int &LDA, int *IPIV, std::complex<double> *B, const int &LDB, int &INFO);
    void zgetrs_(const char &TRANS, const int &N, const int &NRHS, const std::complex<double> *A, const int &LDA, 
            int *IPIV, std::complex<double> *B, const int &LDB, int &INFO);

    void dgetrf_(const int &M, const int &N, double *A, const int &LDA, int *IPIV, int &INFO);
    void dgbtrf_(const int &M, const int &N, const int &KL, const int &KU, double *A, const int &LDA, int *IPIV, int &INFO);
    void dgbtrs_(const char &TRANS, const int &N, const int &KL, const int &KU, const int &NRHS, 
            const double *A, const int &LDA, int *IPIV, double *B, const int &LDB, int &INFO);
    void dgetrs_(const char &TRANS, const int &N, const int &NRHS, const double *A, const int &LDA, int *IPIV, double *B, const int &LDB, int &INFO);
}


inline int max(int a, int b) { return (a > b) ? a : b;}
inline int min(int a, int b) { return (a < b) ? a : b;}


// class for band matrix working with lapack
// square matrix only
template<typename _Scalar, int KL, int KU, int _Cols>
class BandMatrix {
    public:
        typedef Eigen::Matrix<_Scalar, 2*KL+KU+1, _Cols> raw_matrix_t;
        typedef _Scalar Scalar;
        enum {
            RowsAtCompileTime = 2*KL+KU+1,
            ColsAtCompileTime = _Cols
        };

        BandMatrix() : mat() {}
        BandMatrix(const int &N) : mat(2*KL+KU+1, N) {}
        BandMatrix(const BandMatrix &b) : mat(b.mat) {}

        void resize(const int &N) { mat.resize(2*KL+KU+1, N); }
        void fill(const Scalar &v) { mat.fill(v); }

        int supers() const { return KU; }
        int subs() const { return KL; }
        int cols() const { return mat.cols(); }
        int rows() const { return mat.cols(); }
        
        void print_raw_mat(std::ostream &os) const { os << mat << std::endl;}

        void to_dense_matrix(Eigen::Matrix<_Scalar, _Cols, _Cols> &dense_mat)
        {
            int N = mat.cols();
            dense_mat.resize(N, N);
            dense_mat.fill(0);
            for (int j = 0; j < N; ++j)
                for (int i = max(0, j-KU); i <= min(N-1, j+KL); ++i)
                    dense_mat(i,j) = mat(KL+KU+i-j, j);
        }

        inline const _Scalar& operator()(const int &i, const int &j) const 
        {
            int N = mat.cols();
            if (max(0, j-KU) <= i && min(N-1, j+KL) >= i) {
                return mat(KL+KU+i-j, j);
            } else
                throw "Out of band of the band matrix";
        }

        inline _Scalar& operator()(const int &i, const int &j) 
        {
            int N = mat.cols();
            if (max(0, j-KU) <= i && min(N-1, j+KL) >= i) {
                return mat(KL+KU+i-j, j);
            } else
                throw "Out of band of the band matrix";
        }
        
        inline const _Scalar& raw_index(const int &i, const int &j) const { return mat(i, j); }
        inline _Scalar& raw_index(const int &i, const int &j) { return mat(i, j); }
        inline _Scalar *data() { return mat.data(); }
        inline const _Scalar *data() const { return mat.data(); }


    private:
        raw_matrix_t mat;
};

template<typename _Scalar, int KL, int KU, int _Cols>
inline std::ostream &operator<<(std::ostream &os, const BandMatrix<_Scalar, KL, KU, _Cols> &M)
{
    int N = M.cols();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            if (i <= min(N-1, j+KL) && i >= max(0, j-KU))
                os << M(i, j) <<  " ";
            else
                os << "0" << " ";
        os << std::endl; 
    }
    return os;
}


// number traits
template<typename T>
struct n_traits {};

template<>
struct n_traits<std::complex<double> > {
    static void xgetrf_(const int &M, const int &N, std::complex<double> *A, const int &LDA, int *IPIV, int &INFO) {
        zgetrf_(M, N, A, LDA, IPIV, INFO);
    }
    static void xgbtrf_(const int &M, const int &N, const int &KL, const int &KU, 
            std::complex<double> *A, const int &LDA, int *IPIV, int &INFO) {
        zgbtrf_(M, N, KL, KU, A, LDA, IPIV, INFO);
    }
    static void xgbtrs_(const char &TRANS, const int &N, const int &KL, const int &KU, const int &NRHS, 
            const std::complex<double> *A, const int &LDA, int *IPIV, std::complex<double> *B, const int &LDB, int &INFO) {
        zgbtrs_(TRANS, N, KL, KU, NRHS, A, LDA, IPIV, B, LDB, INFO);
    }
    static void xgetrs_(const char &TRANS, const int &N, const int &NRHS, const std::complex<double> *A, const int &LDA, 
            int *IPIV, std::complex<double> *B, const int &LDB, int &INFO) {
        zgetrs_(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO);
    }
};

template<>
struct n_traits<double> {
    static void xgetrf_(const int &M, const int &N, double *A, const int &LDA, int *IPIV, int &INFO) {
        dgetrf_(M, N, A, LDA, IPIV, INFO);
    }
    static void xgbtrf_(const int &M, const int &N, const int &KL, const int &KU, double *A, const int &LDA, int *IPIV, int &INFO) {
        dgbtrf_(M, N, KL, KU, A, LDA, IPIV, INFO);
    }
    static void xgbtrs_(const char &TRANS, const int &N, const int &KL, const int &KU, const int &NRHS, 
            const double *A, const int &LDA, int *IPIV, double *B, const int &LDB, int &INFO) {
        dgbtrs_(TRANS, N, KL, KU, NRHS, A, LDA, IPIV, B, LDB, INFO);
    }
    static void xgetrs_(const char &TRANS, const int &N, const int &NRHS, const double *A, const int &LDA, 
            int *IPIV, double *B, const int &LDB, int &INFO) {
        dgetrs_(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO);
    }
};


// matrix traits
template<class matrix_t>
struct matrix_traits {};

// general dense matrix
template<class T, int NN>
struct matrix_traits<Eigen::Matrix<T, NN, NN> > {
    static void xxxtrf(const int &N, Eigen::Matrix<T, NN, NN> &A, Eigen::VectorXi &IPIV, int &INFO) {
        n_traits<T>::xgetrf_(N, N, A.data(), N, IPIV.data(), INFO);
    }
    static void xxxtrs(const int &N, const Eigen::Matrix<T, NN, NN> &A, Eigen::Matrix<T, NN, NN> &B, Eigen::VectorXi &IPIV, int &INFO) {
        static const char TRANS = 'N';
        n_traits<T>::xgetrs_(TRANS, N, N, A.data(), N, IPIV.data(), B.data(), N, INFO);
    }
};

// band matrix
template<typename T, int KL, int KU, int _Cols>
struct matrix_traits<BandMatrix<T, KL, KU, _Cols> > {
    static void xxxtrf(const int &N, BandMatrix<T, KL, KU, _Cols> &A, Eigen::VectorXi &IPIV, int &INFO) {
        static const int LDA = 2*KL+KU+1;
        n_traits<T>::xgbtrf_(N, N, KL, KU, A.data(), LDA, IPIV.data(), INFO);
    }
    static void xxxtrs(const int &N, const BandMatrix<T, KL, KU, _Cols> &A, Eigen::Matrix<T, _Cols, _Cols> &B, 
            Eigen::VectorXi &IPIV, int &INFO) {
        static const int LDA = 2*KL+KU+1;
        static const char TRANS = 'N';
        n_traits<T>::xgbtrs_(TRANS, N, KL, KU, N, A.data(), LDA, IPIV.data(), B.data(), N, INFO);
    }
};


// main routine
template<class matrix_t>
void inverse(matrix_t &mat, Eigen::Matrix<typename matrix_t::Scalar, matrix_t::ColsAtCompileTime, matrix_t::ColsAtCompileTime> &result)
{
    if (mat.rows() != mat.cols())
        throw "This code works for square matrix only.";
    int N = mat.rows(), INFO;
    Eigen::VectorXi IPIV(N);

    matrix_traits<matrix_t>::xxxtrf(N, mat, IPIV, INFO);
    if (INFO != 0)
        throw "LU decomposition failed.";

    result.resize(N, N);
    result.setIdentity(N, N);
    matrix_traits<matrix_t>::xxxtrs(N, mat, result, IPIV, INFO);
    if (INFO != 0)
        throw "Inversion failed.";
}



// for sparse matrix
#ifdef EIGEN_UMFPACK_SUPPORT
template<class matrix_t>
void sparse_inverse(const Eigen::SparseMatrix<typename matrix_t::Scalar> &mat, matrix_t &result)
{
    int N = mat.rows();
    if (mat.cols() != N)
        throw "This code works for square matrix only.";

    if ((matrix_t::ColsAtCompileTime != Eigen::Dynamic) && (matrix_t::ColsAtCompileTime != N))
        throw "Size of input and output matrices not matched.";

    Eigen::SparseLU<Eigen::SparseMatrix<typename matrix_t::Scalar>, Eigen::UmfPack> lu(mat);
    if(!lu.succeeded()) 
        throw "Sparse matrix: LU decomposition failed.";

    matrix_t B;
    B.setIdentity(N, N);
    result.resize(N, N);
    if (!lu.solve(B, &result))
        throw "Inversion failed.";
}

#endif // EIGEN_UMFPACK_SUPPORT 

} // end namespace inv

#endif // __INVERSE_H__
