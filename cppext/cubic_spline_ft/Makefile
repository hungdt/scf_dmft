# $(ALPS_HOME) is the installation directory of ALPS
include $(ALPS_HOME)/share/alps/include.mk

include ../../Make.inc

CXXFLAGS += -I$(PYTHON_INC) -I$(EIGEN) -I$(NUMPY) -I$(MYINC) -fPIC
LIBS += $(PYTHON_LIB) -lboost_python

# compile mesh classes
TARGET = fourier.so

OBJECTS = main.o fouriertransform.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared $(OBJECTS) $(LDFLAGS) $(LIBS) -o $(TARGET)
    
clean:
	-rm -f *.o $(TARGET)
	
