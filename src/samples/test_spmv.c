#include "mmio_highlevel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <spmv.h>
#include <zconf.h>
#include <wait.h>
int cmp_s(const void*a,const void *b){
    double *s = a;
    double *ss = b;
    if(*s>*ss)return 1;
    else if(*ss>*s)return -1;
    else return 0;
}
// sum up 8 single-precision numbers
void testForFunctions(const char *matrixName,
                      int threads_begin, int threads_end,
                      VALUE_TYPE *Y_golden,
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
    if(FUNC_WAY==Method_Serial) {
        threads_begin = 1;
        threads_end = 1;
    }
    int nnzR = RowPtr[m] - RowPtr[0];
    struct timeval t1, t2;
    int currentiter = 0;
    spmv_Handle_t handle = NULL;


    //qsort(Y_golden,m,sizeof(VALUE_TYPE),cmp_s);
    for(BASIC_SIZE_TYPE thread = threads_begin; thread <= threads_end ; thread<<=1u) {
        omp_set_num_threads(thread);
        gettimeofday(&t1, NULL);
        spmv_create_handle_all_in_one(&handle, m, n, RowPtr, ColIdx, Matrix_Val,
                                      thread, FUNC_WAY, sizeof(VALUE_TYPE), PRODUCT_WAY);
        gettimeofday(&t2, NULL);
        double time =((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) ;


        gettimeofday(&t1, NULL);
        for (currentiter = 0; currentiter < 100; currentiter++) {
            spmv(handle, m, RowPtr, ColIdx, Matrix_Val, Vector_Val_X, Vector_Val_Y);
        }

        gettimeofday(&t2, NULL);
        int iter = 100 +
                10000/(((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0));
        gettimeofday(&t1, NULL);

        for (currentiter = 0; currentiter < iter; currentiter++) {
            spmv(handle, m, RowPtr, ColIdx, Matrix_Val, Vector_Val_X, Vector_Val_Y);
        }

        gettimeofday(&t2, NULL);

        double time_overall_serial =
                ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / iter;
        double GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
        double s = 0;
        //qsort(Vector_Val_Y,m,sizeof(VALUE_TYPE),cmp_s);
        for(int ind = 0 ; ind < m ; ++ind){
            s+=(Vector_Val_Y[ind]-Y_golden[ind])/m*
                    (Vector_Val_Y[ind]-Y_golden[ind])/m;
            //if(fabs(Vector_Val_Y[ind]-Y_golden[ind])>0.001){
                //printf("%d %f %f\n",ind,Vector_Val_Y[ind],Y_golden[ind]);
            //}
        }
        s = sqrt(s);
       // printf("Matrix,Methods,Vectorized,threads,error,predeal-time,time,Gflops\n");
        printf("%s,%s,%s,%lu,%d,%f,%f,%f,%f\n",matrixName,
               Methods_names[FUNC_WAY], Vectorized_names[PRODUCT_WAY], thread,nnzR, s,
               time,time_overall_serial, GFlops_serial);

        if (handle)
            spmv_destory_handle(handle);
    }
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
    //for(int i = 0 ; i < *nnzR ; ++i)(*Val)[i] = 1;
    for (int i = 0; i < *n; i++)
        (*X)[i] = rand()%8*0.125;;

    for (int i = 0; i < *m; i++) {
        for (int j = (*RowPtr)[i]; j < (*RowPtr)[i + 1]; j++)
            (*Y_Golden)[i] += (*Val)[j] * (*X)[(*ColIdx)[j]];
    }

}
#ifndef TEST_METHOD
#define TEST_METHOD Method_Total_Size
#endif
int main(int argc, char ** argv) {

    char *file = argv[1];
    int threads_bregin = atoi(argv[2]);
    int threads_end = atoi(argv[3]);

    BASIC_INT_TYPE m,n,nnzR,isS;
    BASIC_INT_TYPE *RowPtr,*ColIdx;
    VALUE_TYPE *Val;
    VALUE_TYPE *X,*Y,*Y_golden;
    //printf("#iter is %i \n", iter);
    LoadMtx_And_GetGolden(file,&m,&n,&nnzR,&isS,&RowPtr,&ColIdx,&Val,&X,&Y_golden,&Y);
    struct timeval t1, t2;
    SPMV_METHODS d = Method_Total_Size;
    VECTORIZED_WAY way[3] = {VECTOR_NONE, VECTOR_AVX2, VECTOR_AVX512};

    testForFunctions(file, threads_bregin, threads_end , Y_golden, m, n, RowPtr, ColIdx, Val, X, Y,
                     VECTOR_AVX2, Method_Parallel );
    testForFunctions(file, threads_bregin, threads_end, Y_golden, m, n, RowPtr, ColIdx, Val, X, Y,
                     VECTOR_AVX2, Method_Balanced );
    testForFunctions(file, threads_bregin, threads_end, Y_golden, m, n, RowPtr, ColIdx, Val, X, Y,
                     VECTOR_AVX2, Method_Balanced2 );
    testForFunctions(file, threads_bregin, threads_end, Y_golden, m, n, RowPtr, ColIdx, Val, X, Y,
                     VECTOR_AVX2, Method_SellCSigma );
    testForFunctions(file, threads_bregin, threads_end, Y_golden, m, n, RowPtr, ColIdx, Val, X, Y,
                     VECTOR_AVX2, Method_CSR5SPMV );

    free(Val);
    free(RowPtr);
    free(ColIdx);
    free(X);
    free(Y);
    free(Y_golden);
}
