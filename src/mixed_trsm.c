#include "mixed_trsm.h"

#define CUSTOM_GEMM 1


void flat_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    int i, j, ii, jj, nt;
    double alpha = 1.0;
    nt = size / tile_size;

    double* aa = ( double* ) malloc( sizeof( double ) * size * size );
    double* bb = ( double* ) malloc( sizeof( double ) * size * size );

    for ( i = 0; i < nt; i++ )
    {
        for ( j = 0; j < nt; j++ )
        {
            for ( ii = 0; ii < tile_size; ii++ )
            {
                for ( jj = 0; jj < tile_size; jj++ )
                {
                    aa[(i*tile_size+ii)*size + (j*tile_size + jj)] = tiled_matrix_a[i*nt + j][ii * tile_size + jj];
                    bb[(i*tile_size+ii)*size + (j*tile_size + jj)] = tiled_matrix_b [i*nt + j][ii * tile_size + jj];
                }
            }
        }
    }

    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, size, size, alpha, aa, size, bb, size );

    for ( i = 0; i < nt; i++ )
    {
        for ( j = 0; j < nt; j++ )
        {
            for ( ii = 0; ii < tile_size; ii++ )
            {
                for ( jj = 0; jj < tile_size; jj++ )
                {
                    tiled_matrix_a[i*nt + j][ii * tile_size + jj] = aa[(i*tile_size+ii)*size + (j*tile_size + jj)];
                    tiled_matrix_b[i*nt + j][ii * tile_size + jj] = bb[(i*tile_size+ii)*size + (j*tile_size + jj)];
                }
            }
        }
    }

    free(aa);
    free(bb);

}


void tile_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni ) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size, tile_size, -1.0, tiled_matrix_a[mi*nt + ki], tile_size, tiled_matrix_b[ki*nt + nni], tile_size, 1, tiled_matrix_b[mi*nt + nni], tile_size );
                    }
                }
            }
        }
    } //End omp master
}


// THis function attempts to use GPU offloading.
void tile_gpu_dtrsm(cublasHandle_t handle, const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    #if CUDA_USABLE == 1
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;
    double mone = -1.0;

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni ) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        double* TILE_A_GPU = tiled_matrix_a[mi*nt + ki];
                        double* TILE_B_GPU = tiled_matrix_b[ki*nt + nni];
                        double* TILE_C_GPU = tiled_matrix_b[mi*nt + nni];

                        //printf("Before target enter data: TILE_C_GPU[1]=%lf\n", TILE_C_GPU[1]);
                        //#pragma omp single
                        #pragma omp target enter data map(to:TILE_A_GPU[0:tile_size*tile_size],TILE_B_GPU[0:tile_size*tile_size],TILE_C_GPU[0:tile_size*tile_size])
                        #pragma omp target data use_device_ptr(TILE_A_GPU,TILE_B_GPU,TILE_C_GPU)
                        {   
                            // May have messed this up. Need to do tests i gueess
                            int cublas_error = cublasDgemm( handle, CUBLAS_OP_T, CUBLAS_OP_T, tile_size, tile_size, tile_size, &mone, TILE_A_GPU, tile_size, TILE_B_GPU, tile_size, &alpha, TILE_C_GPU, tile_size );

                            if( cublas_error != CUBLAS_STATUS_SUCCESS )
                            {
                                fprintf(stderr, "CUBLAS failed: %d.\n", cublas_error);
                                exit(1);
                            }
        
                        }
                        // wait for call to finish
                        cudaDeviceSynchronize();
                        //printf("Before target exit data: TILE_C_GPU[1]=%lf\n", TILE_C_GPU[1]);
                        #pragma omp target exit data map(release:TILE_A_GPU[0:tile_size*tile_size],TILE_B_GPU[0:tile_size*tile_size]) map(from:TILE_C_GPU[0:tile_size*tile_size])
                        //printf("After target enter data: TILE_C_GPU[1]=%lf\n", TILE_C_GPU[1]);
                    }
                }
            }
        }
    } //End omp master
    #else
    fprintf(stderr, "CUDA not loaded: tile_gpu_dtrsm() has no effect\n");
    #endif
}


void tile_strsm(const int size, const int tile_size, double** dtiled_matrix_a, double** dtiled_matrix_b)
{
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;

    float** tiled_matrix_a = ( float** ) malloc( sizeof( float* ) * nt * nt );
    float** tiled_matrix_b = ( float** ) malloc( sizeof( float* ) * nt * nt );

    for (int i = 0; i < nt; i++ )
    {
        for (int j = 0; j < nt; j++ )
        {
            tiled_matrix_a[i*nt + j] = ( float* ) malloc( sizeof( float ) * tile_size * tile_size );
            tiled_matrix_b[i*nt + j] = ( float* ) malloc( sizeof( float ) * tile_size * tile_size );
            for (int ii = 0; ii < tile_size; ii++ )
            {
                for (int jj = 0; jj < tile_size; jj++ )
                {
                    tiled_matrix_a[i*nt + j][ii*tile_size+ jj] = (float)dtiled_matrix_a[i*nt + j][ii*tile_size+ jj];
                    tiled_matrix_b[i*nt + j][ii*tile_size+ jj] = (float)dtiled_matrix_b[i*nt + j][ii*tile_size+ jj];
                }
            }
        }
    }

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_strsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, tile_size, tile_size, tile_size, -1, tiled_matrix_a[mi*nt + ki], tile_size, tiled_matrix_b[ki*nt + nni], tile_size, 1, tiled_matrix_b[mi*nt + nni], tile_size );
                    }
                }
            }
        }
    } //End omp master

    for (int i = 0; i < nt; i++ )
    {
        for (int j = 0; j < nt; j++ )
        {
            for (int ii = 0; ii < tile_size; ii++ )
            {
                for (int jj = 0; jj < tile_size; jj++ )
                {
                    dtiled_matrix_b[i*nt + j][ii*tile_size+ jj] = (double)tiled_matrix_b[i*nt + j][ii*tile_size+ jj];
                }
            }
            free(tiled_matrix_a[i*nt + j]);
            free(tiled_matrix_b[i*nt + j]);
        }
    }
    free(tiled_matrix_a);
    free(tiled_matrix_b);
}


void tile_sgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        float* f_tile_a1 = (float *)malloc( sizeof(float) * tile_size * tile_size );
                        float* f_tile_b1 = (float *)malloc( sizeof(float) * tile_size * tile_size );
                        float* f_tile_c1 = (float *)calloc( tile_size * tile_size, sizeof(float) );

                        for (int l = 0; l < tile_size*tile_size; l++)
                        {
                            f_tile_a1[l] = (float)(tiled_matrix_a[mi*nt + ki][l]);
                            f_tile_b1[l] = (float)(tiled_matrix_b[ki*nt + nni][l]);
                        }

                        i_sgemm(f_tile_a1, f_tile_b1, f_tile_c1, tile_size, tile_size, tile_size);
                        
                        for (int l = 0; l < tile_size*tile_size; l++)
                        {
                            tiled_matrix_b[mi*nt + nni][l] -= (double)(f_tile_c1[l]);
                        }

                        free(f_tile_a1);
                        free(f_tile_b1);
                        free(f_tile_c1);
                    }
                }
            }
        }
    } //End omp master
}


void tile_dsgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        double* tile_c = (double *)calloc( tile_size * tile_size, sizeof(double) );

                        i_dsgemm(tiled_matrix_a[mi*nt + ki], tiled_matrix_b[ki*nt + nni], tile_c, tile_size, tile_size, tile_size);
                        
                        for (int l = 0; l < tile_size*tile_size; l++)
                        {
                            tiled_matrix_b[mi*nt + nni][l] -= tile_c[l];
                        }

                        free(tile_c);
                    }
                }
            }
        }
    } //End omp master
}

void tile_4dsgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        double* tile_c = (double *)calloc( tile_size * tile_size, sizeof(double) );

                        i_4dsgemm(tiled_matrix_a[mi*nt + ki], tiled_matrix_b[ki*nt + nni], tile_c, tile_size, tile_size, tile_size);
                        
                        for (int l = 0; l < tile_size*tile_size; l++)
                        {
                            tiled_matrix_b[mi*nt + nni][l] -= tile_c[l];
                        }

                        free(tile_c);
                    }
                }
            }
        }
    } //End omp master
}


void tile_3dsgemm_dtrsm(const int size, const int tile_size, double** tiled_matrix_a, double** tiled_matrix_b)
{
    int ki, ni, nni, mi, nt;
    double alpha, beta;
    nt = size / tile_size;

    #pragma omp parallel
    #pragma omp master
    {
        for ( ki = 0; ki < nt; ki++ )
        {
            if ( ki == 0 )
            {
                alpha = 1.0;
            }
            else
            {
                alpha = 1.0;
            }

            for ( ni = 0; ni < nt; ni++ )
            {
                #pragma omp task firstprivate( ki, ni ) depend( in:tiled_matrix_a[ki*nt + ki] ) depend( inout:tiled_matrix_b[ki*nt + ni] )
                {
                    cblas_dtrsm( CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, tile_size, tile_size, alpha, tiled_matrix_a[ki*nt + ki], tile_size, tiled_matrix_b[ki*nt + ni], tile_size );
                }
            }
            for ( nni = 0; nni < nt; nni++ )
            {
                for ( mi = ki + 1; mi < nt; mi++ )
                {
                    #pragma omp task firstprivate( ki, mi, nni) depend( in:tiled_matrix_a[mi*nt + ki] ) depend( in:tiled_matrix_b[ki*nt + nni] ) depend( inout:tiled_matrix_b[mi*nt + nni] )
                    {
                        double* tile_c = (double *)calloc( tile_size * tile_size, sizeof(double) );

                        i_3dsgemm(tiled_matrix_a[mi*nt + ki], tiled_matrix_b[ki*nt + nni], tile_c, tile_size, tile_size, tile_size);
                        
                        for (int l = 0; l < tile_size*tile_size; l++)
                        {
                            tiled_matrix_b[mi*nt + nni][l] -= tile_c[l];
                        }

                        free(tile_c);
                    }
                }
            }
        }
    } //End omp master
}