include ../Make.inc

CXXFLAGS += -I$(PYTHON_INC) -I$(EIGEN) -I$(NUMPY) -I$(MYINC) -fPIC -fopenmp -DEIGEN_DONT_PARALLELIZE
LIBS += $(PYTHON_LIB) -lboost_python -lgomp $(LAPACK)

NAME = bethe_lattice
TARGET = int_$(NAME).so

OBJECTS = main.o integrand.o

all: process_main $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared $(OBJECTS) $(LIBS) -o $(TARGET)

copy: $(TARGET)
	-cp -f $(TARGET) ..

clean:
	-rm -f *.o $(TARGET)

process_main:
	@(\
	  if ! grep "BOOST_PYTHON_MODULE(int_$(NAME))" main.cpp >> /dev/null ; then \
		rm -f *.o *.so; \
		sed -i 's/^BOOST_PYTHON_MODULE.*/BOOST_PYTHON_MODULE(int_$(NAME))/' main.cpp; \
	fi) ;

.PHONY: process_main
