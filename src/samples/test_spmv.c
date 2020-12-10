#include "mmio_highlevel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gemv.h>
// sum up 8 single-precision numbers
void testForFunctions(const char *functionName,
                      int iter, int nthreads,
                      const BASIC_VAL_TYPE *Y_golden,
                      BASIC_INT_TYPE m,
                      const BASIC_INT_TYPE*RowPtr,
                      const BASIC_INT_TYPE *ColIdx,
                      const BASIC_VAL_TYPE*Matrix_Val,
                      const BASIC_VAL_TYPE*Vector_Val_X,
                      BASIC_VAL_TYPE*Vector_Val_Y,
                      VECTORIZED_WAY PRODUCT_WAY,
                      STATUS_GEMV_HANDLE FUNC_WAY
) {
    int nnzR = RowPtr[m] - RowPtr[0];
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    int currentiter = 0;
    gemv_Handle_t handle;
    spmv_create_handle_all_in_one(&handle,m,RowPtr,ColIdx,Matrix_Val,
                                  nthreads,FUNC_WAY,sizeof(BASIC_VAL_TYPE),PRODUCT_WAY);


    for (currentiter = 0; currentiter < iter; currentiter++) {
        spmv(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y);
    }
    gettimeofday(&t2, NULL);
    BASIC_VAL_TYPE time_overall_serial = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
    BASIC_VAL_TYPE GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
    int errorcount_serial = 0;
    for (int i = 0; i < m; i++)
        if (Vector_Val_Y[i] != Y_golden[i])
            errorcount_serial++;

    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=serial-=-=-=-=-=-=-=-=-=-=-=-=-=-\n");
    //printf("time_overall_serial = %f\n", time_overall_serial);
    printf("ErrorCount_%s = %i\n", functionName, errorcount_serial);
    printf("GFlops_%s = %f\n", functionName, GFlops_serial);
    if (handle)
        gemv_destory_handle(handle);
}

int main(int argc, char ** argv) {
    //freopen("out.txt","w",stdout); //输出重定向，输出数据将保存在out.txt文件中
    char *filename = argv[1];
    printf("filename = %s\n", filename);
#ifdef AVC
    printf("ssss\n");
#endif
    //read matrix
    int m, n, nnzR, isSymmetric;

    mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
    int *RowPtr = (int *) aligned_alloc(ALIGENED_SIZE,(m + 1) * sizeof(int));
    int *ColIdx = (int *) aligned_alloc(ALIGENED_SIZE,nnzR * sizeof(int));
    BASIC_VAL_TYPE *Val = (BASIC_VAL_TYPE *) aligned_alloc(ALIGENED_SIZE, nnzR * sizeof(BASIC_VAL_TYPE));
    mmio_data(RowPtr, ColIdx, Val, filename);
    for (int i = 0; i < nnzR; i++)
        Val[i] = 1;
    printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n", m, n, nnzR);

    //create X, Y,Y_golden
    BASIC_VAL_TYPE *X = (BASIC_VAL_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(BASIC_VAL_TYPE) * (n));
    BASIC_VAL_TYPE *Y = (BASIC_VAL_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(BASIC_VAL_TYPE) * (m));
    BASIC_VAL_TYPE *Y_golden = (BASIC_VAL_TYPE *) malloc(sizeof(BASIC_VAL_TYPE) * (m));

    memset(X, 0, sizeof(BASIC_VAL_TYPE) * (n));
    memset(Y, 0, sizeof(BASIC_VAL_TYPE) * (m));
    memset(Y_golden, 0, sizeof(BASIC_VAL_TYPE) * (m));

    for (int i = 0; i < n; i++)
        X[i] = 1;

    for (int i = 0; i < m; i++)
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
            Y_golden[i] += Val[j] * X[ColIdx[j]];

    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);
    printf("#threads is %i \n", nthreads);

    int iter = atoi(argv[3]);
    //printf("#iter is %i \n", iter);

//------------------------------------serial--------------------------------
    struct timeval t1, t2;
     STATUS_GEMV_HANDLE d = STATUS_TOTAL_SIZE;
    VECTORIZED_WAY way[3] = {VECTOR_NONE, VECTOR_AVX2, VECTOR_AVX512};
    for (int i = 6; i < STATUS_TOTAL_SIZE*VECTOR_TOTAL_SIZE; ++i) {
        testForFunctions(funcNames[i], iter, nthreads, Y_golden, m, RowPtr, ColIdx, Val, X, Y,
                         i%VECTOR_TOTAL_SIZE, i/VECTOR_TOTAL_SIZE);
    }

    //free(X);
    free(Y);
    free(Y_golden);
}