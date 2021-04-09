//
// Created by kouushou on 2021/2/4.
//

#include "mmio_highlevel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>

int cmp_s(const void *a, const void *b) {
    double *s = a;
    double *ss = b;
    if (*s > *ss)return 1;
    else if (*ss > *s)return -1;
    else return 0;
}

void mem_flush(const void *p, unsigned int allocation_size) {
    const size_t cache_line = 64;
    const char *cp = (const char *) p;
    size_t i = 0;

    if (p == NULL || allocation_size <= 0)
        return;

    for (i = 0; i < allocation_size; i += cache_line) {
        asm volatile("clflush (%0)\n\t"
        :
        : "r"(&cp[i])
        : "memory");
    }

    asm volatile("sfence\n\t"
    :
    :
    : "memory");
}

static const size_t LLC_CAPACITY = 32 * 1024 * 1024;
static const double *bufToFlushLlc = NULL;

void flushLlc() {
    double sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < LLC_CAPACITY / sizeof(bufToFlushLlc[0]); ++i) {
        sum += bufToFlushLlc[i];
    }
    FILE *fp = fopen("/dev/null", "w");
    fprintf(fp, "%f\n", sum);
    fclose(fp);
}

#define BASIC_INT_TYPE int
#define ALIGENED_SIZE 64
#define BASIC_SIZE_TYPE unsigned int

// sum up 8 single-precision numbers
void testForFunctions(const char *matrixName,
                      int threads_begin, int threads_end,
                      VALUE_TYPE *Y_golden,
                      BASIC_INT_TYPE m,
                      BASIC_INT_TYPE n,
                      BASIC_INT_TYPE *RowPtr,
                      BASIC_INT_TYPE *ColIdx,
                      VALUE_TYPE *Matrix_Val,
                      VALUE_TYPE *Vector_Val_X,
                      VALUE_TYPE *Vector_Val_Y
) {
    int nnzR = RowPtr[m] - RowPtr[0];
    struct timeval t1, t2;
    int currentiter = 0;
    VALUE_TYPE *YY = aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * m);
    VALUE_TYPE *XX = aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * n);

    sparse_matrix_t mat;
    mkl_sparse_d_create_csr(&mat, 0, m, n, RowPtr, RowPtr + 1, ColIdx, Matrix_Val);
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    descr.mode = SPARSE_FILL_MODE_FULL;
    //qsort(Y_golden,m,sizeof(VALUE_TYPE),cmp_s);
    for (BASIC_SIZE_TYPE thread = threads_begin; thread <= threads_end; thread <<= 1u) {

        omp_set_num_threads(thread);
        omp_set_dynamic(0);
        mkl_set_num_threads(thread);
        mkl_set_dynamic(0);

        gettimeofday(&t1, NULL);

        mkl_sparse_set_mv_hint(mat,
                               SPARSE_OPERATION_NON_TRANSPOSE, descr, 5000
        );

        gettimeofday(&t2, NULL);
        double time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);


        memcpy(XX, Vector_Val_X, sizeof(VALUE_TYPE) * n);

        gettimeofday(&t1, NULL);
        for (currentiter = 0; currentiter < 1000; currentiter++) {
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mat, descr, XX, 0, YY);
        }

        gettimeofday(&t2, NULL);
        int iter = 100 +
                   1000 / (((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0));
        iter = 200;
        double time_overall_serial = 0;
        double time_min = 1e9;
        double time_cur;
        //clearFlush();
        for (currentiter = 0; currentiter < iter; currentiter++) {
            mkl_set_num_threads(thread);
            mkl_set_dynamic(0);
            gettimeofday(&t1, NULL);
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mat, descr, XX, 0, YY);
            gettimeofday(&t2, NULL);
            time_cur =
                    ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
            time_overall_serial += time_cur;
            time_min = time_min > time_cur ? time_cur : time_min;
        }
        time_overall_serial /= iter;
        double GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
        double GFlops_Fastest = 2 * nnzR / time_min / pow(10, 6);
        double s = 0;
        //qsort(Vector_Val_Y,m,sizeof(VALUE_TYPE),cmp_s);


        memcpy(Vector_Val_Y, YY, sizeof(VALUE_TYPE) * m);

        for (int ind = 0; ind < m; ++ind) {
            s += (Vector_Val_Y[ind] - Y_golden[ind]) / m *
                 (Vector_Val_Y[ind] - Y_golden[ind]);
            //if(fabs(Vector_Val_Y[ind]-Y_golden[ind])>0.001){
            //printf("%d %f %f\n",ind,Vector_Val_Y[ind],Y_golden[ind]);
            //}
        }
        s = sqrt(s);
        // printf("Matrix,Methods,Vectorized,threads,error,predeal-time,time,Gflops\n");
        printf("%s,%s,%s,%lu,%d,%f,%f,%f,%f,%f\n", matrixName,
               "MKL", "MKL", thread, nnzR, s,
               time, time_overall_serial, GFlops_serial, GFlops_Fastest);

    }
    free(XX);
    free(YY);
}

void LoadMtx_And_GetGolden(char *filePath,
                           BASIC_INT_TYPE *m, BASIC_INT_TYPE *n, BASIC_INT_TYPE *nnzR, BASIC_INT_TYPE *isSymmetric,
                           BASIC_INT_TYPE **RowPtr, BASIC_INT_TYPE **ColIdx, VALUE_TYPE **Val,
                           VALUE_TYPE **X, VALUE_TYPE **Y_Golden, VALUE_TYPE **Y

) {

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    if (mmio_read_from_bin(m, n, nnzR, RowPtr, ColIdx, Val, filePath, ALIGENED_SIZE)) {

        int *RowPtrCpy, *ColIdxCpy;
        VALUE_TYPE *valTypeCpy;
        int ret = mmio_allinone(m, n, nnzR, isSymmetric, &RowPtrCpy, &ColIdxCpy, &valTypeCpy, filePath);
        if (ret)exit(1);
        mmio_save_as_bin(*m, *n, *nnzR, RowPtrCpy, ColIdxCpy, valTypeCpy, filePath);

        *RowPtr = (int *) aligned_alloc(ALIGENED_SIZE, (*m + 1) * sizeof(int));
        *ColIdx = (int *) aligned_alloc(ALIGENED_SIZE, *nnzR * sizeof(int));
        *Val = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, *nnzR * sizeof(VALUE_TYPE));
        memcpy(*RowPtr, RowPtrCpy, (*m + 1) * sizeof(int));
        memcpy(*ColIdx, ColIdxCpy, *nnzR * sizeof(int));
        memcpy(*Val, valTypeCpy, *nnzR * sizeof(VALUE_TYPE));

        //mmio_data(*RowPtr, *ColIdx, *Val, filePath);
        free(RowPtrCpy);
        free(ColIdxCpy);
        free(valTypeCpy);
    }
    //int ret = mmio_info(m, n, nnzR, isSymmetric, filePath);
    gettimeofday(&t2, NULL);

    double t = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
    //create X, Y,Y_golden
    *X = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * (*n));
    *Y = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * (*m));
    *Y_Golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (*m));

    memset(*X, 0, sizeof(VALUE_TYPE) * (*n));
    memset(*Y, 0, sizeof(VALUE_TYPE) * (*m));
    memset(*Y_Golden, 0, sizeof(VALUE_TYPE) * (*m));
    srand(*m);
    for (int i = 0; i < *nnzR; ++i)(*Val)[i] = rand() % 8 * 0.125;
    for (int i = 0; i < *n; i++)
        (*X)[i] = rand() % 8 * 0.125;;

    for (int i = 0; i < *m; i++) {
        for (int j = (*RowPtr)[i]; j < (*RowPtr)[i + 1]; j++)
            (*Y_Golden)[i] += (*Val)[j] * (*X)[(*ColIdx)[j]];
    }

}

#ifndef TEST_METHOD
#define TEST_METHOD Method_Total_Size
#endif

int main(int argc, char **argv) {
    if (argc != 4)return 1;
    char *file = argv[1];
    int threads_bregin = atoi(argv[2]);
    int threads_end = atoi(argv[3]);

    BASIC_INT_TYPE m, n, nnzR, isS;
    BASIC_INT_TYPE *RowPtr, *ColIdx;
    VALUE_TYPE *Val;
    VALUE_TYPE *X, *Y, *Y_golden;
    //printf("#iter is %i \n", iter);
    LoadMtx_And_GetGolden(file, &m, &n, &nnzR, &isS, &RowPtr, &ColIdx, &Val, &X, &Y_golden, &Y);
    struct timeval t1, t2;
    testForFunctions(file, threads_bregin, threads_end, Y_golden, m, n, RowPtr, ColIdx, Val, X, Y);

    free(Val);
    free(RowPtr);
    free(ColIdx);
    free(X);
    free(Y);
    free(Y_golden);
}
