PYTHON = /usr/include/python2.6
EIGEN = /opt/include/eigen3
MY_INC = $(HOME)/works/code/include
NUMPY = /usr/lib64/python2.6/site-packages/numpy/core/include

CXXFLAGS = -I$(PYTHON) -I$(EIGEN) -I$(MY_INC) -I$(NUMPY) -fPIC -O3 -msse4a -fopenmp -DEIGEN_DONT_PARALLELIZE
LIBS = -lboost_python -lpython2.6 -lgomp -L/opt/acml/gfortran64/lib -lacml -lacml_mv

NAME = 2dhopping
TARGET = int_$(NAME).so

OBJECTS = main.o integrand.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared -Wl --export-dynamic $(OBJECTS) $(LIBS) -o $(TARGET)
    
clean:
	-rm -f *.o $(TARGET)
