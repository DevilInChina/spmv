#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <spmv.h>
#include "mmio_highlevel.h"



void printmat(VALUE_TYPE *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%4.8f ", A[i * n + j]);
        printf("\n");
    }
}

void printvec(VALUE_TYPE *x, int n)
{
    for (int i = 0; i < n; i++)
        printf("%4.8f\n", x[i]);
}

VALUE_TYPE vec2norm(VALUE_TYPE *x, int n)
{
    VALUE_TYPE sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}
float vec2norm_float(float *x,int n)
{
   float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

void matvec(VALUE_TYPE *A, VALUE_TYPE *x, VALUE_TYPE *y, int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0;
        for (int j = 0; j < n; j++)
            y[i] += A[i * n + j] * x[j];
    }
}


VALUE_TYPE dotproduct(VALUE_TYPE *x1, VALUE_TYPE *x2, int n)
{
    VALUE_TYPE sum = 0;
    for (int i = 0; i < n; i++)
        sum += x1[i] * x2[i];
    return sum;
}
float dotproduct_float(float *x1,float *x2,int n)
{
    float  sum=0;
    for (int i = 0; i < n; i++)
        sum += x1[i] * x2[i];
    return sum;
}
void cg_VALUE_TYPE(int *RowPtr,int *ColIdx,VALUE_TYPE *Val,
               VALUE_TYPE *x, VALUE_TYPE *b, int n, int *iter, int maxiter, VALUE_TYPE threshold)
{
    memset(x, 0, sizeof(VALUE_TYPE) * n);
    VALUE_TYPE *residual = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    VALUE_TYPE *y = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    VALUE_TYPE *p = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    VALUE_TYPE *q = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    *iter = 0;
    VALUE_TYPE norm = 0;
    VALUE_TYPE rho = 0;
    VALUE_TYPE rho_1 = 0;
    spmv_Handle_t temp = NULL;
    spmv_create_handle_all_in_one(&temp, n, RowPtr, ColIdx, Val, 8, Method_SellCSigma, sizeof(Val[0]), VECTOR_AVX512);
    // p0 = r0 = b - Ax0
    //matvec(A, x, y, n);
    spmv(temp,n,RowPtr,ColIdx,Val,x,y);
    for (int i = 0; i < n; i++)
        residual[i] = b[i] - y[i];
    //printvec(residual, n);

    do
    {
        //printf("\niter = %i\n", *iter);
        rho = dotproduct(residual, residual, n);
        if (*iter == 0)
        {
            for (int i = 0; i < n; i++)
                p[i] = residual[i];
        }
        else
        {
            VALUE_TYPE beta = rho / rho_1;
            for (int i = 0; i < n; i++)
                p[i] = residual[i] + beta * p[i];
        }

        //matvec(A, p, q, n);
        spmv(temp,n,RowPtr,ColIdx,Val,p,q);
        VALUE_TYPE alpha = rho / dotproduct(p, q, n);
        //printf("alpha = %f\n", alpha);
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];
        for (int i = 0; i < n; i++)
            residual[i] += - alpha * q[i];

        rho_1 = rho;
        VALUE_TYPE error = vec2norm(residual, n); // / vec2norm(b, n);
        printf("error=%f\n",error);
        //printvec(x, n);
        *iter += 1;

        if (error < threshold)
            break;
    }
    while (*iter < maxiter);

    free(residual);
    free(y);
    free(p);
    free(q);
}
int main(int argc, char **argv)
{
    //int n;
    VALUE_TYPE *A, *x, *b;

    // method: gauss, lu_doolittle, cholesky
    printf("\n");
   
    char *filename = argv[1];
    printf ("filename = %s\n", filename);
    //read matrix_float
//    int m, n, nnzR, isSymmetric;
//    mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
//    int *RowPtr = (int *) aligned_alloc(ALIGENED_SIZE,(m + 1) * sizeof(int));
//    int *ColIdx = (int *) aligned_alloc(ALIGENED_SIZE,nnzR * sizeof(int));
//    float *Val = (float *) aligned_alloc(ALIGENED_SIZE, nnzR * sizeof(float));
//    mmio_data(RowPtr, ColIdx, Val, filename);
//    float *X = (float *) aligned_alloc(ALIGENED_SIZE, sizeof(float) * (n));
//    float *Y = (float *) aligned_alloc(ALIGENED_SIZE, sizeof(float) * (m));
//    float *Y_golden = (float *) malloc(sizeof(float) * (m));
//    memset(X, 0, sizeof(float) * (n));
//    memset(Y, 0, sizeof(float) * (m));
//    memset(Y_golden, 0, sizeof(float) * (m));
//    for(int i = 0; i < n; i++)
//    X[i] = 1;
//    int iter=0;
//    for (int i = 0; i < m; i++)
//        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
//            Y_golden[i] += Val[j] * X[ColIdx[j]];
//        new_cg_float(RowPtr,ColIdx,Val,X,Y_golden,n,&iter,1000,0.00001);
//    printf("\n#iter of CG = %i\n", iter);
    //read_matrix_VALUE_TYPE
    int m, n, nnzR, isSymmetric;
    mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
    int *RowPtr = (int *) aligned_alloc(ALIGENED_SIZE,(m + 1) * sizeof(int));
    int *ColIdx = (int *) aligned_alloc(ALIGENED_SIZE,nnzR * sizeof(int));
    VALUE_TYPE *Val = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, nnzR * sizeof(VALUE_TYPE));
    mmio_data(RowPtr, ColIdx, Val, filename);
    VALUE_TYPE *X = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * (n));
    VALUE_TYPE *Y = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, sizeof(VALUE_TYPE) * (m));
    VALUE_TYPE *Y_golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (m));
    memset(X, 0, sizeof(VALUE_TYPE) * (n));
    memset(Y, 0, sizeof(VALUE_TYPE) * (m));
    memset(Y_golden, 0, sizeof(VALUE_TYPE) * (m));
    for(int i = 0; i < n; i++)
        X[i] = 1;
    int iter=0;
    for (int i = 0; i < m; i++)
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
            Y_golden[i] += Val[j] * X[ColIdx[j]];
    cg_VALUE_TYPE(RowPtr,ColIdx,Val,X,Y_golden,n,&iter,1000,0.00001);
    printf("\n#iter of CG = %i\n", iter);
//    free(A);
//    free(x);
//    free(b);
}
