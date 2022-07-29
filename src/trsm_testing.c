#include "mixed_trsm.h"


// This is intended to be called by python file numerical_analysis.py
int run_trsm_tests(int size, int tile_size, double** TILE_A, double** D_TILE_B, double** SS_TILE_B, double** S_TILE_B , double** M_TILE_B, int which_precision)
{
    flat_dtrsm(size, tile_size, TILE_A, D_TILE_B);
    tile_strsm(size, tile_size, TILE_A, SS_TILE_B);
    tile_sgemm_dtrsm(size, tile_size, TILE_A, S_TILE_B);
    if(which_precision==1)
    {
        tile_dsgemm_dtrsm(size, tile_size, TILE_A, M_TILE_B);
    }
    else if (which_precision==2)
    {
        tile_4dsgemm_dtrsm(size, tile_size, TILE_A, M_TILE_B);
    }
    else if (which_precision==3)
    {
        tile_3dsgemm_dtrsm(size, tile_size, TILE_A, M_TILE_B);
    }
    return 0;
}


int generate_random_matrices(int size, int tile_size, int rand_seed, double** TILE_A, double** TILE_B)
{
    int i, j, ii, jj, nt;
    double rand_val, rand_val2;
    srand(rand_seed);

    nt = size / tile_size;

    double* aa = ( double* ) malloc( sizeof( double ) * size * size );
    double* bb = ( double* ) malloc( sizeof( double ) * size * size );

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            rand_val = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX));
            rand_val2 = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX));
            aa[i*size + j] = (double)(rand() % 3) + 1.0 + rand_val;
            bb[i*size + j] = (double)(rand() % 3) + 1.0;

            if (i == j)
            {
                aa[i*size + j] += 20;
            }
        }
    }

    for ( i = 0; i < nt; i++ )
    {
        for ( j = 0; j < nt; j++ )
        {
            for ( ii = 0; ii < tile_size; ii++ )
            {
                for ( jj = 0; jj < tile_size; jj++ )
                {
                    if ((i*tile_size + ii) - (j*tile_size + jj) >= 0)
                        TILE_A[i*nt + j][ii * tile_size + jj] = aa[(i*tile_size+ii)*size + (j*tile_size + jj)];
                    else
                        TILE_A[i*nt + j][ii * tile_size + jj] = 0;

                    TILE_B [i*nt + j][ii * tile_size + jj] = bb[(i*tile_size+ii)*size + (j*tile_size + jj)];
                }
            }
        }
    }
    free(aa);
    free(bb);

    return 0;
}


int generate_random_matrices_from_X(int size, int tile_size, int rand_seed, double** TILE_A, double** TILE_B, double** X)
{
    int i, j, ii, jj, nt;
    double rand_val, rand_val2;
    srand(rand_seed);

    nt = size / tile_size;

    double* aa = ( double* ) malloc( sizeof( double ) * size * size );
    double* bb = ( double* ) malloc( sizeof( double ) * size * size );
    double* xx = ( double* ) malloc( sizeof( double ) * size * size );

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            rand_val = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX));
            rand_val2 = ((double)(rand())+(double)(rand())) / (2.0*(double)(RAND_MAX));
            xx[i*size + j] = rand_val2*100.0 - 50;
            if (i-j > 0)
                aa[i*size + j] = rand_val*15.0 + 5;
            else if (i-j == 0)
                aa[i*size + j] = rand_val*15.0 + 55;
            else
                aa[i*size + j] = 0;
        }
    }

    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, aa, size, xx, size, 0, bb, size );

    for ( i = 0; i < nt; i++ )
    {
        for ( j = 0; j < nt; j++ )
        {
            for ( ii = 0; ii < tile_size; ii++ )
            {
                for ( jj = 0; jj < tile_size; jj++ )
                {
                    if ((i*tile_size + ii) - (j*tile_size + jj) >= 0)
                        TILE_A[i*nt + j][ii * tile_size + jj] = aa[(i*tile_size+ii)*size + (j*tile_size + jj)];
                    else
                        TILE_A[i*nt + j][ii * tile_size + jj] = 0;

                    TILE_B[i*nt + j][ii * tile_size + jj] = bb[(i*tile_size+ii)*size + (j*tile_size + jj)];

                    X[i*nt + j][ii * tile_size + jj] = xx[(i*tile_size+ii)*size + (j*tile_size + jj)];
                }
            }
        }
    }
    free(aa);
    free(bb);
    free(xx);

    return 0;
}


int generate_random_matrices_lapacke(int size, int tile_size, int rand_seed, double** TILE_A, double** TILE_B)
{
    if (size > 35000)
    {
        fprintf(stderr, "VERY Large matrix, requested operation not compelted, consider not using generate_random_matrices_lapacke! Modify code to continue.\n");
        return 1;
    }

    int i, j, ii, jj, nt;
    double rand_val, rand_val2, alpha;
    srand(rand_seed);

    nt = size / tile_size;
    alpha = ( double ) rand() / ( double ) rand() + DBL_MIN;

    double* aa = ( double* ) malloc( sizeof( double ) * size * size );
    double* bb = ( double* ) malloc( sizeof( double ) * size * size );
    lapack_int * ipiv = ( lapack_int * ) malloc( size * sizeof( lapack_int  ) ); // Something is off here. using size* sizeof( int ) gives a seg fault.
    
    LAPACKE_DLARNV(1, &rand_seed, size * size, aa);
    /*
    while (cond_num < 0.001)
    {
        LAPACKE_DLARNV(1, &rand_seed, size * size, aa);
        org_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', size, size, aa, size);

        LAPACKE_dgetrf(LAPACK_COL_MAJOR, size, size, aa, size, ipiv);
        LAPACKE_dgecon(LAPACK_COL_MAJOR, '1', size, aa, size, org_norm, &cond_num);
        //printf("Condition number: %lf\n", cond_num);
        rand_seed++;
    }
    */

    LAPACKE_dgetrf(LAPACK_COL_MAJOR, size, size, aa, size, ipiv);

    LAPACKE_DLARNV(1, &rand_seed, size * size, bb);

    for ( i = 0; i < nt; i++ )
    {
        for ( j = 0; j < nt; j++ )
        {
            for ( ii = 0; ii < tile_size; ii++ )
            {
                for ( jj = 0; jj < tile_size; jj++ )
                {
                    if ((i*tile_size + ii) - (j*tile_size + jj) >= 0)
                        TILE_A[i*nt + j][ii * tile_size + jj] = alpha * aa[(i*tile_size+ii)*size + (j*tile_size + jj)];
                    else
                        TILE_A[i*nt + j][ii * tile_size + jj] = 0;

                    TILE_B [i*nt + j][ii * tile_size + jj] = alpha * bb[(i*tile_size+ii)*size + (j*tile_size + jj)];
                }
            }
        }
    }
    free(aa);
    free(bb);
    free(ipiv);

    return 0;
}


int run_cpugpu_gflops_tests(int max_r, int init_size, int nt, int multiplier, int rand_seed)
{
    #if CUDA_USABLE == 1
    int size, tile_size, r, i , j;
    double flops, cpu_flops_iris, gpu_flops_iris, time;
    struct timeval start, end;
    double** mat_a;
    double** mat_cpu_b;
    double** mat_gpu_b;
    double** mat_gpu_bp;
    int error = 0;
    FILE *fptr;
    
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization failed!\n");

        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "Offloading devices found: %d\n", omp_get_num_devices());
    if (omp_get_num_devices() < 1)
    {
        fprintf(stderr, "Potential Error with GPU offloading(no devices found).\n");
    }

    fptr = fopen("./output_file","w");

    mat_a = (double **)malloc(sizeof(double*) * nt * nt);
    mat_cpu_b = (double **)malloc(sizeof(double*) * nt * nt);
    mat_gpu_b = (double **)malloc(sizeof(double*) * nt * nt);
    mat_gpu_bp = (double **)malloc(sizeof(double*) * nt * nt);
    
    size = init_size;

    for (r = 0; r < max_r; r++)
    {
        fprintf(stdout, "Computing with size=%d\n", size);
        flops = FLOPS_DTRSM(size, size);
        tile_size = size / nt;
        
        /************************
         *  generate matricies  *
         ************************/
        for (i = 0; i< nt; i++)
        {
            for (j = 0; j< nt; j++)
            {
                mat_a[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
                mat_cpu_b[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
                mat_gpu_b[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
                mat_gpu_bp[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
            }
        }

        generate_random_matrices_lapacke(size, tile_size, rand_seed, mat_a, mat_cpu_b);

        for (i = 0; i< nt; i++)
        {
            for (j = 0; j< nt; j++)
            {
                for (int l = 0; l < tile_size*tile_size; l++)
                {
                    mat_gpu_b[i*nt + j][l] = mat_cpu_b[i*nt + j][l];
                    mat_gpu_bp[i*nt + j][l] = mat_cpu_b[i*nt + j][l];
                }
            }
        }

        /***************
         *  RUN tests  *
         ***************/

        // CPU only
        gettimeofday(&start, NULL);

        tile_dtrsm(size, tile_size, mat_a, mat_cpu_b);

        gettimeofday(&end, NULL);
        time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);
        cpu_flops_iris = flops / time;


        //CPU+GPU
        tile_gpu_dtrsm(handle, size, tile_size, mat_a, mat_gpu_bp);
        gettimeofday(&start, NULL);

        tile_gpu_dtrsm(handle, size, tile_size, mat_a, mat_gpu_b);

        gettimeofday(&end, NULL);
        time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);
        gpu_flops_iris = flops / time;

        /*********************************
         *  Error checking and printing  *
         *********************************/
        //error checking
        for ( int i = 0; i < nt; i++ )
        {
            for ( int j = 0; j < nt; j++ )
            { 
                for ( int ii = 0; ii < tile_size * tile_size; ii++ )
                {
                    if( fabs( mat_cpu_b[i*nt + j][ii] - mat_gpu_b[i*nt + j][ii] ) > fabs( mat_cpu_b[i*nt + j][ii] * 0.00001) && error < 10)
                    {
                        fprintf(stderr, "ERROR: CPU[%d][%d][%d] = %f, GPU[%d][%d][%d] = %f\n", i, j, ii, mat_cpu_b[i*nt + j][ii], i, j, ii, mat_gpu_b[i*nt + j][ii] );
                        error++;
                    }
                }
            }
        }

        // Printing to stdout
        fprintf(stdout, "CPU_DTRSM_GFLOPS=%.4g, CPUGPU_DTRSM_GFLOPS=%.4g\n", cpu_flops_iris / 1e9, gpu_flops_iris / 1e9 );
        // Printing to output file
        fprintf(fptr, "%.17g,%.17g\n", cpu_flops_iris / 1e9, gpu_flops_iris / 1e9);

        // Free memory
        for (i = 0; i< nt; i++)
        {
            for (j = 0; j< nt; j++)
            {
                free(mat_a[i*nt + j]);
                free(mat_cpu_b[i*nt + j]);
                free(mat_gpu_b[i*nt + j]);
                free(mat_gpu_bp[i*nt + j]);
            }
        }
        if (error >= 10)
        {
            fprintf(stderr, "10 errors occured in computation. stopping tests\n");
            break;
        }
        size *= multiplier;
    }
    cublasDestroy(handle);

    fclose(fptr);

    free(mat_a);
    free(mat_cpu_b);
    free(mat_gpu_b);
    free(mat_gpu_bp);

    return 0;
    #else
    fprintf(stderr, "CUDA libraries not loaded. Exiting\n" );
    return 1;
    #endif
}


int run_cpugpu_gflops_test(int size, int tile_size, int rand_seed)
{
    #if CUDA_USABLE == 1
    int i , j;
    double flops, cpu_flops_iris, gpu_flops_iris, time;
    struct timeval start, end;
    double** mat_a;
    double** mat_cpu_b;
    double** mat_gpu_b;
    double** mat_gpu_bp;
    int error = 0;
    int nt = size / tile_size;
    
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization failed!\n");

        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "Offloading devices found: %d\n", omp_get_num_devices());
    if (omp_get_num_devices() < 1)
    {
        fprintf(stderr, "Potential error with GPU offloading(no devices found).\n");
        //return 1;
    }

    mat_a = (double **)malloc(sizeof(double*) * nt * nt);
    mat_cpu_b = (double **)malloc(sizeof(double*) * nt * nt);
    mat_gpu_b = (double **)malloc(sizeof(double*) * nt * nt);
    mat_gpu_bp = (double **)malloc(sizeof(double*) * nt * nt);

    fprintf(stdout, "Computing with size=%d\n", size);
    flops = FLOPS_DTRSM(size, size);
    
    /************************
     *  generate matricies  *
     ************************/
    for (i = 0; i< nt; i++)
    {
        for (j = 0; j< nt; j++)
        {
            mat_a[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
            mat_cpu_b[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
            mat_gpu_b[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
            mat_gpu_bp[i*nt + j] = (double*)malloc(sizeof(double) * tile_size * tile_size);
        }
    }

    generate_random_matrices_lapacke(size, tile_size, rand_seed, mat_a, mat_cpu_b);

    for (i = 0; i< nt; i++)
    {
        for (j = 0; j< nt; j++)
        {
            for (int l = 0; l < tile_size*tile_size; l++)
            {
                mat_gpu_b[i*nt + j][l] = mat_cpu_b[i*nt + j][l];
                mat_gpu_bp[i*nt + j][l] = mat_cpu_b[i*nt + j][l];
            }
        }
    }

    /***************
     *  RUN tests  *
     ***************/

    // CPU only
    gettimeofday(&start, NULL);

    tile_dtrsm(size, tile_size, mat_a, mat_cpu_b);

    gettimeofday(&end, NULL);
    time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);
    cpu_flops_iris = flops / time;


    //CPU+GPU
    
    tile_gpu_dtrsm(handle, size, tile_size, mat_a, mat_gpu_bp);
    tile_gpu_dtrsm(handle, size, tile_size, mat_a, mat_gpu_bp);
    
    gettimeofday(&start, NULL);
    tile_gpu_dtrsm(handle, size, tile_size, mat_a, mat_gpu_b);
    gettimeofday(&end, NULL);
    time = ( double ) (((end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec)) / 1e6);
    gpu_flops_iris = flops / time;

    /*********************************
     *  Error checking and printing  *
     *********************************/
    //error checking
    for ( int i = 0; i < nt; i++ )
    {
        for ( int j = 0; j < nt; j++ )
        { 
            for ( int ii = 0; ii < tile_size * tile_size; ii++ )
            {
                if( fabs( mat_cpu_b[i*nt + j][ii] - mat_gpu_b[i*nt + j][ii] ) > fabs( mat_cpu_b[i*nt + j][ii] * 0.00001) && error < 10)
                {
                    fprintf(stderr, "ERROR: CPU[%d][%d][%d] = %f, GPU[%d][%d][%d] = %f\n", i, j, ii, mat_cpu_b[i*nt + j][ii], i, j, ii, mat_gpu_b[i*nt + j][ii] );
                    error++;
                }
            }
        }
    }

    // Printing to stdout
    fprintf(stdout, "CPU_DTRSM_GFLOPS=%.4g, CPUGPU_DTRSM_GFLOPS=%.4g\n", cpu_flops_iris / 1e9, gpu_flops_iris / 1e9 );

    // Free memory
    for (i = 0; i< nt; i++)
    {
        for (j = 0; j< nt; j++)
        {
            free(mat_a[i*nt + j]);
            free(mat_cpu_b[i*nt + j]);
            free(mat_gpu_b[i*nt + j]);
            free(mat_gpu_bp[i*nt + j]);
        }
    }
    if (error >= 10)
    {
        fprintf(stderr, "10 errors occured in computation. stopping tests\n");
    }

    cublasDestroy(handle);

    free(mat_a);
    free(mat_cpu_b);
    free(mat_gpu_b);
    free(mat_gpu_bp);

    return 0;
    #else
    fprintf(stderr, "CUDA libraries not loaded. Exiting\n" );
    return 1;
    #endif
}


int main()
{
    
    //fprintf(stdout, "Running indivdual tests\n");
    //run_cpugpu_gflops_test(256, 32, 1);
    //run_cpugpu_gflops_test(4096, 128, 1);
    //run_cpugpu_gflops_test(4096, 512, 1);

    
    fprintf(stdout, "Running tests and generating to file\n");
    // Will run for 11 iterations starting at size=16 to 16348 with tile_size=size/8
    run_cpugpu_gflops_tests(4, 32, 8, 2, 1);

    return 0;
}
