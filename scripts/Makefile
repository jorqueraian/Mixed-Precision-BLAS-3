# Summit
SUMMIT_CPU_LIB     = -L$(OLCF_ESSL_ROOT)/lib64
SUMMIT_CPU_INC     = -I$(OLCF_ESSL_ROOT)/include
SUMMIT_COMP_LIB    = -L$(OLCF_XL_ROOT)/xlsmp/5.1.1/lib
SUMMIT_COMP2_LIB   = -L$(OLCF_XL_ROOT)/xlf/16.1.1/lib
SUMMIT_RUN_LIB     = -R$(OLCF_XL_ROOT)/lib

SUMMIT_GPU_LIB     = -L$(OLCF_CUDA_ROOT)/lib64
SUMMIT_GPU_INC     = -I$(OLCF_CUDA_ROOT)/include
# ESSL libs on summit: libessl.so       libessl6464.so       libesslsmp.so       libesslsmp6464.so       libesslsmpcuda.so
SUMMIT_CXXFLAGS = -lesslsmp -lxlf90_r -lxlsmp -lxlfmath $(SUMMIT_CPU_INC) $(SUMMIT_CPU_LIB) $(SUMMIT_COMP_LIB) $(SUMMIT_COMP2_LIB) $(SUMMIT_RUN_LIB) $(SUMMIT_GPU_INC) $(SUMMIT_GPU_LIB) -O3 -qsmp=omp -qoffload -lcuda -lcudart -lcublas -pedantic 
SUMMIT_CXXFLAGS_SERIAL = -lessl -lxlf90_r -lxlfmath $(SUMMIT_CPU_INC) $(SUMMIT_CPU_LIB) $(SUMMIT_COMP_LIB) $(SUMMIT_COMP2_LIB) $(SUMMIT_RUN_LIB) $(SUMMIT_GPU_INC) $(SUMMIT_GPU_LIB) -O3 -qsmp=omp -qoffload -lcuda -lcudart -lcublas -pedantic 
SUMMIT_CC = xlc_r


l: 
	xlc_r -qpic -shared -o libmixedtrsm.so ../src/trsm_testing.c ../src/mixed_trsm.c -lesslsmp6464 -qsmp=omp

e: 
	gcc -fpic -shared -o libmixedtrsm.so ../src/trsm_testing.c ../src/mixed_trsm.c ../src/mixed_gemm.c -I /opt/intel/oneapi/mkl/2022.1.0/include -L /opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -L /opt/intel/oneapi/compiler/2022.1.0/linux/compiler/lib/intel64 -L /opt/intel/oneapi/intelpython/python3.9/lib -O3 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl -fopenmp -liomp5 -lquadmath

c:
	gcc -fpic -shared -o libmixedtrsm.so ../src/trsm_testing.c ../src/mixed_trsm.c -I /opt/AMD/aocl/aocl-linux-gcc-3.0-6/include /opt/AMD/aocl/aocl-linux-gcc-3.0-6/lib/libblis.a -L /opt/AMD/aocl/aocl-linux-gcc-3.0-6/lib -lm -lpthread -fopenmp
s: 
	xlc_r -qpic -shared -o libmixedtrsm.so ../src/trsm_testing.c ../src/mixed_trsm.c $(SUMMIT_CXXFLAGS)
clean:
	rm -rf *.o *.out *.so *.ptx *.cub *.lst *.mod test
