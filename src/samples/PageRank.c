//
// Created by ydc on 2020/12/17.
//
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <spmv.h>
#include "mmio_highlevel.h"
/* Step Two:
   sequential implementation of the PageRank algorithm with
   CSR representation of matrix A
*/
void mv(int n,int *Rowptr,int *ColIndex,double *Value,double *x,double *y)
{
    for(int i=0;i<n;i++)
        y[i]=0.0;
    for(int i=0;i<n;i++)
    {
        for(int j=Rowptr[i];j<Rowptr[i+1];j++)
        {
            int k=ColIndex[j];
            y[i]+=Value[j]*x[k];
        }
    }
}
int main(int argc,char **argv){

    int i,j;
    char *filename = argv[1];
    printf ("filename = %s\n", filename);
    int n, n_csr, nnzR, isSymmetric;
    mmio_info(&n, &n_csr, &nnzR, &isSymmetric, filename);
    /* case: COO formats */
    int *row_ptr = (int *) aligned_alloc(ALIGENED_SIZE,(n + 1) * sizeof(int));
    int *col_ind = (int *) aligned_alloc(ALIGENED_SIZE,nnzR * sizeof(int));
    VALUE_TYPE *val = (VALUE_TYPE *) aligned_alloc(ALIGENED_SIZE, nnzR * sizeof(double));
    mmio_data(row_ptr, col_ind, val, filename);

    // Fix the stochastization
    int *out_link=(int *)malloc(n*sizeof(n));
    for(i=0; i<n; i++){
        out_link[i] =0;
    }


    int rowel = 0;
    for(i=0; i<n; i++){
        if (row_ptr[i+1] != 0) {
            rowel = row_ptr[i+1] - row_ptr[i];
            out_link[i] = rowel;
        }
    }

    int curcol = 0;
    for(i=0; i<n; i++){
        rowel = row_ptr[i+1] - row_ptr[i];
        for (j=0; j<rowel; j++) {
            val[curcol] = val[curcol] / out_link[i];
            curcol++;
        }
    }

    /******************* INITIALIZATION OF P, DAMPING FACTOR ************************/

    // Set the damping factor 'd'
    double d = 0.85;

    // Initialize p[] vector
    double *p=(double *)malloc(n*sizeof(double));
    for(i=0; i<n; i++){
        p[i] = 1.0/n;
    }
    /*************************** PageRank LOOP  **************************/

    // Set the looping condition and the number of iterations 'k'
    int looping = 1;
    int k = 0;

    // Initialize new p vector
    double *p_new=(double *)malloc(sizeof(double)*n);
    while (k<1000){

        // Initialize p_new as a vector of n 0.0 cells
        for(i=0; i<n; i++){
            p_new[i] = 0.0;
        }

        int rowel = 0;
        int curcol = 0;
        spmv_Handle_t balanced_handle;
        int nthreads=8;//线程数
        spmv_create_handle_all_in_one(&balanced_handle,n,n,row_ptr,col_ind,val,nthreads,
                                      Method_Balanced2,sizeof(val[0] ),VECTOR_NONE);
        spmv(balanced_handle,n,row_ptr,col_ind,val,p,p_new);
        // Adjustment to manage dangling elements
        for(i=0; i<n; i++){
            p_new[i] = d * p_new[i] + (1.0 - d) / n;
        }


        // TERMINATION: check if we have to stop
        float error = 0.0;
        for(i=0; i<n; i++) {
            error =  error + fabs(p_new[i] - p[i]);
        }
        printf("iter=%d error=%f\n",k,error);
        //if two consecutive instances of pagerank vector are almost identical, stop
        if (error < 0.000001){
            break;
            looping = 0;
        }

        // Update p[]
        for (i=0; i<n;i++){
            p[i] = p_new[i];
        }

        // Increase the number of iterations
        k = k + 1;
    }
    printf ("\nNumber of iteration to converge: %d \n\n", k);
    printf ("Final Pagerank values:\n\n[");
    return 0;
}


