/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License
    as published by the Free Software Foundation, either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    the Lesser GNU General Public License for more details.

    You should have received a copy of the the Lesser GNU General
    Public License along with this program.  If not, see
    <http://www.gnu.org/licenses/>.
    */

/*
 * Copy and modify the code from the net
 * http://forum.kde.org/viewtopic.php?f=74&t=86057
 * Hung Dang, May 14, 2011
 */ 

#ifndef __NUMPY_EIGEN_H__
#define __NUMPY_EIGEN_H__

#include <numpy/arrayobject.h>
#include <Eigen/Core>

namespace numpy {


template <typename SCALAR>
struct NumpyEquivalentType {};

template <> struct NumpyEquivalentType<double> {enum { type_code = NPY_DOUBLE };};
template <> struct NumpyEquivalentType<int> {enum { type_code = NPY_INT };};
template <> struct NumpyEquivalentType<float> {enum { type_code = NPY_FLOAT };};
template <> struct NumpyEquivalentType<std::complex<double> > {enum { type_code = NPY_CDOUBLE };};

template<typename MatrixType>
struct eigen_python_matrix_converter
{
    typedef Eigen::internal::traits<MatrixType> traits;

    //////////////////////////
    // to python conversion
    static void to_python(const MatrixType& matrix, PyObject *&ret) {
        import_array();
        npy_intp shape[2] = { matrix.rows(), matrix.cols() };
        PyArrayObject* python_array = (PyArrayObject*)PyArray_SimpleNew(2, shape, NumpyEquivalentType<typename traits::Scalar>::type_code);
        copy_array(matrix.data(), (typename traits::Scalar*)PyArray_DATA(python_array), matrix.rows(), matrix.cols());
        Py_INCREF(python_array);
        ret = PyArray_Return(python_array);
    }

    ////////////////////////////
    // from python conversion
    static void convertible(PyObject* obj_ptr){
        if (!PyArray_Check(obj_ptr))
            throw "PyArray_Check failed";
        if (PyArray_NDIM(obj_ptr)!=2)
            throw "dim != 2";
        if (PyArray_ObjectType(obj_ptr, 0) != NumpyEquivalentType<typename traits::Scalar>::type_code) 
            throw "types not compatible";
        int elsize = (PyArray_DESCR(obj_ptr))->elsize;
        if (PyArray_STRIDE(obj_ptr, 0) % elsize != 0 || PyArray_STRIDE(obj_ptr, 1) % elsize != 0)
            throw "strides and type size not matched";
    }

    static void from_python(PyObject* py_obj_ptr, MatrixType &matrix)
    {
        import_array();
        convertible(py_obj_ptr);
        PyArrayObject *array = reinterpret_cast<PyArrayObject*>(py_obj_ptr);

        npy_int nb_rows = array->dimensions[0];
        npy_int nb_cols = array->dimensions[1];
        int elsize = (PyArray_DESCR(py_obj_ptr))->elsize;
        npy_int row_stride = PyArray_STRIDE(py_obj_ptr, 0)/elsize,
                col_stride = PyArray_STRIDE(py_obj_ptr, 1)/elsize;

        matrix.resize(nb_rows, nb_cols);
        copy_array((typename traits::Scalar*)PyArray_DATA(array), matrix.data(), nb_rows, nb_cols, 
                true, traits::Flags & Eigen::RowMajorBit, row_stride, col_stride);
    }

    template <typename SourceType, typename DestType >
        static void copy_array(const SourceType* source, DestType* dest, const npy_int &nb_rows, const npy_int &nb_cols,
                const bool &isSourceTypeNumpy = false, const bool &isDestRowMajor = true, 
                const npy_int &numpy_row_stride = 1, const npy_int &numpy_col_stride = 1) 
        {
            // determine source strides
            int row_stride = 1, col_stride = 1;
            if (isSourceTypeNumpy) {
                row_stride = numpy_row_stride;
                col_stride = numpy_col_stride;
            } else {
                if (traits::Flags & Eigen::RowMajorBit) 
                    row_stride = nb_cols;
                else col_stride = nb_rows;
            }

            if (isDestRowMajor) 
                for (int r=0; r<nb_rows; r++) 
                    for (int c=0; c<nb_cols; c++) {
                        *dest = source[r*row_stride + c*col_stride];
                        dest++;
                    }
            else 
                for (int c=0; c<nb_cols; c++)
                    for (int r=0; r<nb_rows; r++) {
                        *dest = source[r*row_stride + c*col_stride];
                        dest++;
                    }
        }
};



template<typename VectorType>
struct eigen_python_vector_converter
{
    typedef Eigen::internal::traits<VectorType> traits;

    //////////////////////////
    // to python conversion
    static void to_python(const VectorType& vector, PyObject *&py_vector) {
        import_array();
        npy_intp shape = vector.size();
        PyArrayObject* python_array = (PyArrayObject*)PyArray_SimpleNew(1, &shape, NumpyEquivalentType<typename traits::Scalar>::type_code);
        copy_array(vector.data(), (typename traits::Scalar*)PyArray_DATA(python_array), vector.size());
        Py_INCREF(python_array);
        py_vector = PyArray_Return(python_array);
    }

    ////////////////////////////
    // from python conversion
    static void convertible(PyObject* obj_ptr){
        if (!PyArray_Check(obj_ptr))
            throw "PyArray_Check failed";
        if (PyArray_NDIM(obj_ptr) != 1)
            throw "dim != 1";
        if (PyArray_ObjectType(obj_ptr, 0) != NumpyEquivalentType<typename traits::Scalar>::type_code)
            throw "types not compatible";
        if (PyArray_STRIDE(obj_ptr, 0) % (PyArray_DESCR(obj_ptr))->elsize != 0)
            throw "stride and type size not matched";
    }

    static void from_python(PyObject* py_obj_ptr, VectorType &vector)
    {
        import_array();
        convertible(py_obj_ptr);

        PyArrayObject *array = reinterpret_cast<PyArrayObject*>(py_obj_ptr);
        npy_int nb_size = array->dimensions[0];

        vector.resize(nb_size);
        npy_int stride = PyArray_STRIDE(py_obj_ptr, 0) / (PyArray_DESCR(py_obj_ptr))->elsize;
        copy_array((typename traits::Scalar*)PyArray_DATA(array), vector.data(), nb_size, stride);
    }

    template <typename SourceType, typename DestType >
        static void copy_array(const SourceType* source, DestType* dest, const npy_int &nb_size, const npy_int &stride = 1) {
            for (int i = 0; i < nb_size; ++i)
                dest[i] = source[stride*i];
        }
};

template<typename T>
void from_numpy(PyObject *py_obj, Eigen::Matrix<T, Eigen::Dynamic, 1> &ret) {
    eigen_python_vector_converter<Eigen::Matrix<T, Eigen::Dynamic, 1> >::from_python(py_obj, ret);
}    

template<typename T>
void from_numpy(PyObject *py_obj, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &ret) {
    eigen_python_matrix_converter<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >::from_python(py_obj, ret);
}


template<class array_t>
PyObject* to_numpy(const array_t &arr)
{
    PyObject *ret;
    if (arr.rows() == 1 || arr.cols() == 1)
        eigen_python_vector_converter<array_t>::to_python(arr, ret);
    else
        eigen_python_matrix_converter<array_t>::to_python(arr, ret);

    return ret;
}

} // end namespace numpy

#endif
