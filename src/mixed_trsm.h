#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include "mixed_gemm.h"

#ifdef __has_include
    #if __has_include(<essl.h>)
        #include <essl.h>
        #define LAPACKE_DLARNV(_idist, _iseed, _size_n, _out_x) for (int i = 0; i < _size_n; i++) { _out_x[i] = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX)); }
    #elif __has_include("blis.h")
        #include "cblas.h"
        #include "blis.h" // Need to find if blis has this
    #else
        #include "mkl.h"
        #include "mkl_lapacke.h"
        #define LAPACKE_DLARNV(_idist, _iseed, _size_n, _out_x) for (int i = 0; i < _size_n; i++) { _out_x[i] = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX)); } // #define LAPACKE_DLARNV(_idist, _iseed, _size_n, _out_x) LAPACKE_dlarnv(_idist, _iseed, _size_n, _out_x)
    #endif
#else
    #include <essl.h>
    #define LAPACKE_DLARNV(_idist, _iseed, _size_n, _out_x) for (int i = 0; i < _size_n; i++) { _out_x[i] = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX)); }
#endif

#ifdef __has_include
    #if !__has_include("cublas_v2.h")
        #warning "Could not load CUDA. GPU TRSMs not compiled."
        typedef typeof(NULL) cublasHandle_t;
    #elif !__has_include("cuda_runtime_api.h")
        #warning "Could not load CUDA. GPU TRSMs not compiled."
        typedef typeof(NULL) cublasHandle_t;
    #else
        #include "cublas_v2.h"
        #include "cuda_runtime_api.h"
        #define CUDA_USABLE 1
    #endif
#else
    #warning "Could not load CUDA. GPU TRSMs not compiled."
    typedef typeof(NULL) cublasHandle_t;
#endif

// FLOPS calculations
// GEMM
// Number of multiplications in GEMM
#define FMULS_GEMM(m_, n_, k_) ( (m_) * (n_) * (k_) )
// Number of additions in GEMM
#define FADDS_GEMM(m_, n_, k_) ( (m_) * (n_) * (k_) )
// Flops in DGEMM 
#define FLOPS_DGEMM(m_, n_, k_) ( FMULS_GEMM((double)(m_), (double)(n_), \
 	(double)(k_)) + FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )

// TRSM
// Number of multiplications in TRSM
#define FMULS_TRSM_2(m_, n_) ( 0.5 * (n_) * (m_) * ( (m_) + 1 ) )
// Number of additions in TRSM
#define FADDS_TRSM_2(m_, n_) ( 0.5 * (n_) * (m_) * ( (m_) - 1 ) )
// Number of multiplies in TRSM
#define FMULS_TRSM( m_, n_) ( FMULS_TRSM_2((m_), (n_)) )
// Number of additions in TRSM
#define FADDS_TRSM( m_, n_) ( FADDS_TRSM_2((m_), (n_)) )
// Flops in DTRSM
#define FLOPS_DTRSM( m_, n_) ( FMULS_TRSM( (double)(m_), (double)(n_)) + FADDS_TRSM( (double)(m_), (double)(n_)) )


void flat_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);


void tile_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);


void tile_gpu_dtrsm(cublasHandle_t handle, const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);


void tile_strsm(const int size, const int tile_size, double** dtiled_matrix_a, double** dtiled_matrix_b);


void tile_sgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);


void tile_dsgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);


void tile_4dsgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);


void tile_3dsgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b);
