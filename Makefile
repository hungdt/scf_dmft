include Make.inc

all:
	@(\
	for d in $(DIRS); do \
		cd $$d; \
		echo Working in $$d; \
		make copy; \
		cd ..; \
	done) 
	
clean:
	-rm -f *.so *.pyc
	@(\
	for d in $(DIRS); do \
		cd $$d; \
		echo Clean the directory $$d; \
		make clean; \
		cd ..; \
	done) 
       	
