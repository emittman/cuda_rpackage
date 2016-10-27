NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC
R_INC = /home/emittman/src/R-3.3.1/builddir/include
LIBS = -lcublas_device -lcublas -lcurand
TARGET = cudarpackage.so
HST_OBJS = host_fns/cholesky.o host_fns/iter_getter.o
WRP_OBJS = cudarpackage.o
#final SHLIB
$(TARGET): $(WRP_OBJS) $(HST_OBJS)
	nvcc -shared $(NVCC_FLAGS) $^ -I$(R_INC) -I. $(LIBS) -o cudarpackage.so

cudarpackage.o: cudarpackage.cu
	nvcc -dc $(NVCC_FLAGS) $< -I$(R_INC) -o $@ $(LIBS)

host_fns/cholesky.o: host_fns/cholesky.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@ $(LIBS) 

host_fns/iter_getter.o: host_fns/iter_getter.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@

.PHONY: clean
clean:
	rm cudarpackage.so $(WRP_OBJS) $(HST_OBJS)
