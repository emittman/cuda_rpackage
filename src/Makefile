NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC
R_INC = /home/emittman/src/R-3.3.1/builddir/include
LIBS = -lcublas_device -lcublas -lcurand

cudarpackage.so: cudarpackage.o
	nvcc -shared $(NVCC_FLAGS) cudarpackage.o -I$(R_INC) -I. $(LIBS) -o cudarpackage.so

cudarpackage.o: cudarpackage.cu
	nvcc -dc $(NVCC_FLAGS) cudarpackage.cu -I$(R_INC) -o cudarpackage.o $(LIBS)

.PHONY: clean
clean:
	rm cudarpackge.so
	rm cudarpackge.o

