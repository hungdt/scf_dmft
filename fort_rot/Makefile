include ../Make.inc

TARGET = fort_rot.so

all: 
	f2py -c fort_rot.f -m fort_rot --fcompiler=gnu95 --f77flags="-fopenmp -ffixed-line-length-132" -lgomp $(LAPACK)

copy: all
	-cp -f $(TARGET) ..

clean:
	-rm -f $(TARGET)
