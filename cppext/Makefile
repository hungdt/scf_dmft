include ../Make.inc

# this is to get info from the old ALPS for the impurity solver from P. Werner
# I don't include it here, so this line is commented
#include /home/td120143/apps/alps-1.3.5/share/alps/include.mk

CXXFLAGS += -I$(PYTHON_INC) -I$(EIGEN) -I$(NUMPY) -I$(MYINC) -fPIC
LIBS += $(PYTHON_LIB) -lboost_python

# compile mesh classes
TARGET = cppext.so

# the file eval.cpp is for P. Werner's impurity solver, not needed
OBJECTS = main.o fourier.o others.o # eval.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared $(OBJECTS) $(LDFLAGS) $(LIBS) -o $(TARGET)

copy: $(TARGET)
	-cp -f $(TARGET) ..

clean:
	-rm -f *.o $(TARGET)
	
