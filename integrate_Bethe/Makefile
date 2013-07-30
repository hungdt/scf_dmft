PYTHON = /usr/include/python2.6
EIGEN = $(HOME)/apps/include/eigen3
MY_INC = $(HOME)/apps/include/my_inc
NUMPY = /usr/lib64/python2.6/site-packages/numpy/core/include

# location of the Boost Python include files and library
BOOST_INC = $(HOME)/apps/share/boost
BOOST_LIB = $(HOME)/apps/share/boost/stage/lib

CXX=g++
CXXFLAGS = -O3 -march=native -mtune=native -I$(EIGEN) -I$(MY_INC) -I$(PYTHON) -I$(BOOST_INC) -I$(NUMPY) -fPIC -DEIGEN_DONT_PARALLELIZE -fopenmp
LIBS += -L$(BOOST_LIB) -lboost_python -lpython2.6 -lgomp -L$(HOME)/apps/lib/acml/gfortran64/lib -lacml -lacml_mv

NAME = Bethe
TARGET = int_$(NAME).so

OBJECTS = main.o integrand.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared -Wl --export-dynamic $(OBJECTS) $(LIBS) -o $(TARGET)
    
clean:
	-rm -f *.o $(TARGET)

