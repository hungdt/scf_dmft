CXXFLAGS = -O3 -ffast-math -mtune=native

PYTHON = /usr/local_rwth/sw/python/2.7.5/x86_64
PYTHON_INC = $(PYTHON)/include/python2.7
PYTHON_LIB = -L$(PYTHON)/lib -lpython2.7

EIGEN = $(HOME)/apps/include/eigen
NUMPY = /rwthfs/rz/SW/UTIL.common/Python/2.7.5/x86_64/extra/lib/python2.7/site-packages/numpy/core/include
MYINC = ../include

DIRS = cppext fort_rot integrate_dft_mlwf integrate_tight_binding integrate_bethe_lattice

LAPACK = -L/usr/lib64/atlas -llapack -lf77blas -lcblas -latlas
