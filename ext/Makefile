include $(HOME)/apps/alps/intel/share/alps/include.mk

CXX = g++

# location of the Python header files
PYTHON = /usr/include/python2.6
EIGEN = $(HOME)/apps/include/eigen3
MY_INC = $(HOME)/apps/include/my_inc
NUMPY = /usr/lib64/python2.6/site-packages/numpy/core/include

CXXFLAGS += -I$(PYTHON) -I$(EIGEN) -I$(NUMPY) -I$(MY_INC) -fPIC
LIBS += -L/home/7fz/apps/share/boost/stage/lib -lboost_python -lpython2.6

# compile mesh classes
TARGET = cppext.so

OBJECTS = main.o fourier.o eval.o others.o 

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared -Wl --export-dynamic $(OBJECTS) $(LDFLAGS) $(LIBS) -o $(TARGET)
    
clean:
	-rm -f *.o $(TARGET)
	
