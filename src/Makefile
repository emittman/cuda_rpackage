NVCC_FLAGS = -arch=sm_35 -Xcompiler -fPIC
R_INC = /home/emittman/src/R-3.3.1/builddir/include
LIBS = -lcublas_device -lcublas -lcurand -lcuda -lcudart
DL_LIBS = dev_link/libquadform.a
DL_PATH = /home/emittman/cuda_rpackage/src/dev_link
DL_FLAGS = -lquadform
TARGET = cudarpackage.so
HST_OBJS = host_fns/cholesky.o host_fns/iter_getter.o host_fns/summary2.o host_fns/chain.cu
DEV_OBJS = dev_link/quad_form.o dev_link/libquadform.a
WRP_OBJS = cudarpackage.o

#final SHLIB
$(TARGET): $(WRP_OBJS) $(HST_OBJS) $(DL_LIBS)
	nvcc -shared $(NVCC_FLAGS) $(WRP_OBJS) $(HST_OBJS) -o $(TARGET) -I$(R_INC) -I. -L$(DL_PATH) $(LIBS) $(DL_FLAGS)

#wrappers
cudarpackage.o: cudarpackage.cu
	nvcc -c $(NVCC_FLAGS) $< -I$(R_INC) -o $@

#host functions
host_fns/cholesky.o: host_fns/cholesky.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@ $(LIBS) 

host_fns/iter_getter.o: host_fns/iter_getter.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@
	
host_fns/summary2.o: host_fns/summary2.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@
	
host_fns/chain.o: host_fns/chain.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@

#device-linked functions
dev_link/libquadform.a: dev_link/quad_form.o
	nvcc -lib $(NVCC_FLAGS) $< -o $@ -lcublas_device

dev_link/quad_form.o: dev_link/quad_form.cu
	nvcc -dc $(NVCC_FLAGS) $< -o $@ -lcublas_device

#make clean
.PHONY: clean
clean:
	rm -f cudarpackage.so $(WRP_OBJS) $(HST_OBJS) $(DEV_OBJS)
