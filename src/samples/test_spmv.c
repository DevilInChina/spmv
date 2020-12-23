#include "mmio_highlevel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <spmv.h>
// sum up 8 single-precision numbers
void testForFunctions(const char *functionName,
                      int iter, int nthreads,
                          const VALUE_TYPE *Y_golden,
                      BASIC_INT_TYPE m,
                      BASIC_INT_TYPE n,
                      const BASIC_INT_TYPE*RowPtr,
                      const BASIC_INT_TYPE *ColIdx,
                      const VALUE_TYPE*Matrix_Val,
                      const VALUE_TYPE*Vector_Val_X,
                      VALUE_TYPE*Vector_Val_Y,
                      VECTORIZED_WAY PRODUCT_WAY,
                      SPMV_METHODS FUNC_WAY
) {
    if(FUNC_WAY==Method_Serial)nthreads = 1;
    int nnzR = RowPtr[m] - RowPtr[0];
    struct timeval t1, t2;
    int currentiter = 0;
    spmv_Handle_t handle = NULL;
    spmv_create_handle_all_in_one(&handle,m,n,RowPtr,ColIdx,Matrix_Val,
                                  nthreads,FUNC_WAY,sizeof(VALUE_TYPE),PRODUCT_WAY);

    for(BASIC_SIZE_TYPE thread = 1u; thread <= nthreads ; thread<<=1u) {
        omp_set_num_threads(thread);
        gettimeofday(&t1, NULL);

        for (currentiter = 0; currentiter < iter; currentiter++) {
            spmv(handle, m, RowPtr, ColIdx, Matrix_Val, Vector_Val_X, Vector_Val_Y);
        }

        gettimeofday(&t2, NULL);

        VALUE_TYPE time_overall_serial = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
        VALUE_TYPE GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
        VALUE_TYPE s = 0;
        for(int ind = 0 ; ind < m ; ++ind){
            s+=(Vector_Val_Y[ind]-Y_golden[ind])/m*
                    (Vector_Val_Y[ind]-Y_golden[ind])/m;
        }
        s = sqrt(s);
        printf("%s,%s,%lu,%16.10f,%.10f\n",
               Methods_names[FUNC_WAY], Vectorized_names[PRODUCT_WAY], thread, s, GFlops_serial);
    }
    if (handle)
        spmv_destory_handle(handle);

}

void LoadMtx_And_GetGolden(char *filePath,
        BASIC_INT_TYPE*m,BASIC_INT_TYPE*n,BASIC_INT_TYPE*nnzR,BASIC_INT_TYPE*isSymmetric,BASIC_INT_TYPE **RowPtr,BASIC_INT_TYPE ** ColIdx, VALUE_TYPE **Val,
                       VALUE_TYPE ** X,VALUE_TYPE **Y_Golden,VALUE_TYPE ** Y

){



    mmio_info(m, n, nnzR, isSymmetric, filePath);
     *RowPtr = (int *) aligned_alloc(ALIGENED_SIZE,(*m + 1) * sizeof(int));
     *ColIdx = (int *) aligned_alloc(ALIGENED_SIZE,*nnzR * sizeof(int));
     *Val = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, *nnzR * sizeof(VALUE_TYPE));
    mmio_data(*RowPtr,* ColIdx, *Val, filePath);

    //create X, Y,Y_golden
     *X = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * (*n));
     *Y = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * (*m));
     *Y_Golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (*m));

    memset(*X, 0, sizeof(VALUE_TYPE) * (*n));
    memset(*Y, 0, sizeof(VALUE_TYPE) * (*m));
    memset(*Y_Golden, 0, sizeof(VALUE_TYPE) * (*m));
    srand(*m);
    for (int i = 0; i < *n; i++)
        (*X)[i] = rand()%8*0.125;

    for (int i = 0; i < *m; i++)
        for (int j = (*RowPtr)[i]; j < (*RowPtr)[i + 1]; j++)
            (*Y_Golden)[i] += (*Val)[j] * (*X)[(*ColIdx)[j]];

}

int main(int argc, char ** argv) {

    char *file = argv[1];
    int nthreads = atoi(argv[2]);
    int iter = atoi(argv[3]);

    BASIC_INT_TYPE m,n,nnzR,isS;
    BASIC_INT_TYPE *RowPtr,*ColIdx;
    VALUE_TYPE *Val;
    VALUE_TYPE *X,*Y,*Y_golden;
    //printf("#iter is %i \n", iter);
    LoadMtx_And_GetGolden(file,&m,&n,&nnzR,&isS,&RowPtr,&ColIdx,&Val,&X,&Y_golden,&Y);
    struct timeval t1, t2;
     SPMV_METHODS d = Method_Total_Size;
    VECTORIZED_WAY way[3] = {VECTOR_NONE, VECTOR_AVX2, VECTOR_AVX512};
    for (int i = Method_Balanced2 *VECTOR_TOTAL_SIZE + VECTOR_AVX2; i < Method_Total_Size * VECTOR_TOTAL_SIZE; ++i) {
        testForFunctions(funcNames[i], iter, nthreads, Y_golden, m,n, RowPtr, ColIdx, Val, X, Y,
                         i%VECTOR_TOTAL_SIZE, i/VECTOR_TOTAL_SIZE);
    }
    free(Val);
    free(RowPtr);
    free(ColIdx);
    free(X);
    free(Y);
    free(Y_golden);
}
