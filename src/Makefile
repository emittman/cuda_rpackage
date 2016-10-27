NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC
R_INC = /home/emittman/src/R-3.3.1/builddir/include
LIBS = -lcublas_device -lcublas -lcurand
DL_LIBS = -lquadform
TARGET = cudarpackage.so
HST_OBJS = host_fns/cholesky.o host_fns/iter_getter.o
WRP_OBJS = cudarpackage.o

#final SHLIB
$(TARGET): $(WRP_OBJS) $(HST_OBJS)
	nvcc -shared $(NVCC_FLAGS) $^ -I$(R_INC) -I. $(LIBS) $(DL_LIBS) -o cudarpackage.so

#wrappers
cudarpackage.o: cudarpackage.cu
	nvcc -dc $(NVCC_FLAGS) $< -I$(R_INC) -o $@ $(LIBS)

#host functions
host_fns/cholesky.o: host_fns/cholesky.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@ $(LIBS) 

host_fns/iter_getter.o: host_fns/iter_getter.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@

#device-linked functions
dev_link/libquadform.a: dev_link/quad_form.o
	nvcc -lib $(NVCC_FLAGS) $< -o $@ -lcublas_device

dev_link/quad_form.o: dev_link/quad_form.cu
	nvcc -dc $(NVCC_FLAGS) $< -o $@ -lcublas_device

#make clean
.PHONY: clean
clean:
	rm -f cudarpackage.so $(WRP_OBJS) $(HST_OBJS)
