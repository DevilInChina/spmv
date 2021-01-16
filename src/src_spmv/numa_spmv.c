//
// Created by kouushou on 2021/1/9.
//
#include "inner_spmv.h"

#include <numa.h>
#include <math.h>
#include <pthread.h>


typedef struct numa_spmv_parameter
{
    int nthreads, numanodes, m, coreidx, alloc;
    spmv_Handle_t handle;
} numa_spmv_parameter,*numa_spmv_parameter_t;

typedef struct NumaEnvironment {
    int **subrowptrA, **subcolidxA;
    void **subvalA, **X, **Y;
    numa_spmv_parameter_t p ;
    pthread_t* threads;
    pthread_attr_t pthread_custom_attr;
    int numanodes;
    int *subm,*subm_ex,*subX_ex, *subX,*subnnz,*subnnz_ex;
    int PARTS;
}NumaEnvironment,*NumaEnvironment_t;



void inner_exclusive_scan(BASIC_INT_TYPE *input, int length)
{
    if(length == 0 || length == 1)
        return;

    BASIC_INT_TYPE old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

void numaHandleDestory(spmv_Handle_t handle){
    if(handle){
        if(handle->extraHandle && handle->spmvMethod == Method_Numa ){
            NumaEnvironment_t numasVal = handle->extraHandle;
            for(int i = 0 ; i < numasVal->numanodes ; ++i){

                numa_free(numasVal->subrowptrA[i],
                          sizeof(int)*(numasVal->subm[numasVal->p[i].alloc]+1));

                numa_free(numasVal->subcolidxA[i],
                          sizeof(int)*numasVal->subnnz[numasVal->p[i].alloc]);

                numa_free(numasVal->subvalA[i],
                          handle->data_size*numasVal->subnnz[numasVal->p[i].alloc]);

                numa_free(numasVal->X[i],
                          handle->data_size*numasVal->subX[numasVal->p[i].alloc]);

                numa_free(numasVal->Y[i],
                          handle->data_size*numasVal->subX[numasVal->p[i].alloc]);

            }

            free(numasVal->p);
            free(numasVal->subX);
            free(numasVal->subX_ex);
            free(numasVal->subm);
            free(numasVal->subm_ex);
            free(numasVal->subnnz);
            free(numasVal->subnnz_ex);
            free(numasVal->threads);
            pthread_attr_destroy(&numasVal->pthread_custom_attr);
            free(handle->extraHandle);
            handle->extraHandle = NULL;
        }
    }
}


void *spmv_numa(void *arg){
    numa_spmv_parameter *pn = (numa_spmv_parameter*)arg;
    NumaEnvironment_t numasVal = pn->handle->extraHandle;
    int me = pn->alloc;
    numa_run_on_node(me);
    int m = pn->m;
    int nthreads = pn->nthreads;
    int numanodes = pn->numanodes;
    int coreidx = pn->coreidx;
    int eachnumathreads = nthreads/numanodes;
    int task = (m+eachnumathreads-1)/eachnumathreads;//ceil((double)m/(double)eachnumathreads);
    int start = coreidx*task;
    int end = (coreidx+1)*task>m?m:(coreidx+1)*task;
    //printf("numanode %d, coreindex %d, m %d, nthreads %d, eachnumathreads %d, start %d, end %d\n",pn->alloc,coreidx,pn->m,nthreads,eachnumathreads,start,end);
    if(pn->handle->data_size == sizeof(double )) {
        double *val = numasVal->subvalA[me];
        //VALUE_TYPE *x = X[me];
        double *y = numasVal->Y[me];
        int *rpt = numasVal->subrowptrA[me];
        int *col = numasVal->subcolidxA[me];
        for (int u = start; u < end; u++) {
            y[u] = 0;
            for (int j = rpt[u]; j < rpt[u + 1]; j++) {
                int Xpos = col[j] / numasVal->subX[0];
                int remainder = col[j] - numasVal->subX_ex[Xpos];
                y[u] += val[j] * ((double **)(numasVal->X))[Xpos][remainder];
            }
            //if(me==7)
            //printf("y[%d][%d]%.2f\n",me,u,sum);
        }
    }else{
        float *val = numasVal->subvalA[me];
        //VALUE_TYPE *x = X[me];
        float *y = numasVal->Y[me];
        int *rpt = numasVal->subrowptrA[me];
        int *col = numasVal->subcolidxA[me];
        for (int u = start; u < end; u++) {
            y[u] = 0;
            for (int j = rpt[u]; j < rpt[u + 1]; j++) {
                int Xpos = col[j] / numasVal->subX[0];
                int remainder = col[j] - numasVal->subX_ex[Xpos];
                y[u] += val[j] * ((float **)(numasVal->X))[Xpos][remainder];
            }
            //if(me==7)
            //printf("y[%d][%d]%.2f\n",me,u,sum);
        }
    }

}

int numa_spmv_get_handle_Selected(spmv_Handle_t handle,
                                      int PARTS,
                                      BASIC_INT_TYPE m,BASIC_INT_TYPE n,
                                      const BASIC_INT_TYPE *RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const void *Matrix_Val
){
    int numanodes = numa_max_node()+1;
    //if(numanodes==1)return 0;
    handle->extraHandle = malloc(sizeof(NumaEnvironment));
    NumaEnvironment_t numaVal = handle->extraHandle;
    numaVal->numanodes = numanodes;
    numaVal->PARTS = PARTS;
    int nthreads = handle->nthreads;
    int eachnumacores = handle->nthreads/numanodes;
    int *subrowpos = (int *)malloc(sizeof(int) * (PARTS+1));
    int *subm = (int *)malloc(sizeof(int) * PARTS);
    int *subm_ex = (int *)malloc(sizeof(int) * (PARTS+1));
    int *subX = (int *)malloc(sizeof(int) * PARTS);
    int *subX_ex = (int *)malloc(sizeof(int) * (PARTS+1));
    int *subnnz = (int *)malloc(sizeof(int) * PARTS);
    int *subnnz_ex = (int *)malloc(sizeof(int) * (PARTS+1));

    numaVal->subm = subm;
    numaVal->subm_ex = subm_ex;
    numaVal->subX = subX;
    numaVal->subX_ex = subX_ex;
    numaVal->subnnz = subnnz;
    numaVal->subnnz_ex = subnnz_ex;
    int i,j;
    for (i = 0; i < PARTS; i++)
        subrowpos[i] = (ceil((double)m/(double)PARTS))*i>m?m:((ceil((double)m/(double)PARTS))*i);
    subrowpos[PARTS] = m;
    for (i = 0; i < (PARTS-1); i++)
    {
        subX[i] = ceil((double)n/(double)PARTS);
    }
    subX[PARTS-1] = n - subX[0]*(PARTS-1);
    for (i = 0; i < PARTS; i++)
    {
        subX_ex[i] = subX[i];
    }
    inner_exclusive_scan(subX_ex, PARTS + 1);
    for (i = 0; i < PARTS; i++)
    {
        subm[i] = subrowpos[i+1] - subrowpos[i];
        subm_ex[i] = subrowpos[i+1] - subrowpos[i];
    }
    inner_exclusive_scan(subm_ex, PARTS + 1);
    for (i = 0; i < PARTS; i++)
    {
        subnnz[i] = RowPtr[subrowpos[i]+subm[i]]-RowPtr[subrowpos[i]];
        subnnz_ex[i] = subnnz[i];
    }
    inner_exclusive_scan(subnnz_ex, PARTS + 1);
    numaVal->p = (numa_spmv_parameter*)malloc(handle->nthreads * sizeof(numa_spmv_parameter));
    numaVal->threads=(pthread_t *)malloc(handle->nthreads*sizeof(pthread_t));
    //pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&numaVal->pthread_custom_attr);
    for(i = 0; i < handle->nthreads; i++)
    {
        numaVal->p[i].alloc = i%numanodes;
        numaVal->p[i].numanodes = numanodes;
        numaVal->p[i].nthreads = nthreads;
        numaVal->p[i].handle = handle;
    }
    for(i = 0; i < eachnumacores; i++)
    {
        for(j = 0; j < numanodes; j++)
        {
            numaVal->p[i*numanodes+j].coreidx = i;
            numaVal->p[i*numanodes+j].m = subm[j];
        }
    }
    numaVal->subrowptrA = (int **)malloc(sizeof(int *)*nthreads);
    numaVal->subcolidxA = (int **)malloc(sizeof(int *)*nthreads);
    numaVal->subvalA = malloc(sizeof(void *)*nthreads);
    numaVal->X = malloc(sizeof(void *)*nthreads);
    numaVal->Y = malloc(sizeof(void *)*nthreads);

    for(i = 0; i < nthreads; i++)
    {
        numaVal->subrowptrA[i] = numa_alloc_onnode(sizeof(int)*(subm[numaVal->p[i].alloc]+1), numaVal->p[i].alloc);

        numaVal->subcolidxA[i] = numa_alloc_onnode(sizeof(int)*subnnz[numaVal->p[i].alloc], numaVal->p[i].alloc);

        numaVal->subvalA[i] = numa_alloc_onnode(handle->data_size*subnnz[numaVal->p[i].alloc], numaVal->p[i].alloc);

        numaVal->X[i] = numa_alloc_onnode(handle->data_size*subX[numaVal->p[i].alloc], numaVal->p[i].alloc);

        numaVal->Y[i] = numa_alloc_onnode(handle->data_size*subm[numaVal->p[i].alloc], numaVal->p[i].alloc);

    }
    int currentcore = 0;
    int k;
    for(i = 0; i < numanodes; i++) {
        for (j = 0; j < eachnumacores; j++) {
            for (k = 0; k <= subm[i]; k++) {
                currentcore = i + j * eachnumacores;
                if (currentcore < handle->nthreads) {
                    numaVal->subrowptrA[currentcore][k] = RowPtr[subrowpos[i] + k];
                }
            }
        }
        for (j = 0; j < eachnumacores; j++) {
            for (k = 0; k < subnnz[i]; k++) {

                currentcore = i + j * eachnumacores;

                if (currentcore < handle->nthreads) {
                    numaVal->subcolidxA[currentcore][k] = ColIdx[subnnz_ex[i] + k];

                    switch (handle->data_size) {
                        case sizeof(double): {
                            ((double **) numaVal->subvalA)[currentcore][k] = ((double *) Matrix_Val)[subnnz_ex[i] + k];
                        }
                            break;
                        default: {
                            ((float **) numaVal->subvalA)[currentcore][k] = ((float *) Matrix_Val)[subnnz_ex[i] + k];
                        }
                            break;
                    }
                }
            }
        }
    }

    for (i = 0; i < handle->nthreads; i++) {
        if (i % numanodes != 0) {
            int temprpt = numaVal->subrowptrA[i][0];
            for ( j = 0; j <= numaVal->subm[i % numanodes]; j++) {
                numaVal->subrowptrA[i][j] -= temprpt;
            }
        }
    }
    return numanodes;
}


void spmv_numa_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y
){
    NumaEnvironment_t numasVal = handle->extraHandle;
    int numanodes = numasVal->numanodes;
    int eachnumacores = handle->nthreads/numasVal->numanodes;

    for(int i = 0 ; i < numanodes ; ++i){
        for (int j = 0; j < eachnumacores; j++) {
            int currentcore = i + j * eachnumacores;
            if (currentcore < handle->nthreads) {

                memcpy(numasVal->X[currentcore],
                       Vector_Val_X+numasVal->subX_ex[i]*handle->data_size,
                       numasVal->subX[i]*handle->data_size
                       );

            }
        }
    }
    for (int i = 0; i < handle->nthreads; i++)
    {
        pthread_create(numasVal->threads+i, &numasVal->pthread_custom_attr, spmv_numa,
                       (void *)(numasVal->p+i));
    }
    for (int i = 0; i < handle->nthreads; i++)
    {
        pthread_join(numasVal->threads[i], NULL);
    }

    for (int i = 0; i < numasVal->PARTS; i++)
    {
        memcpy(
                Vector_Val_Y+numasVal->subm_ex[i]*handle->data_size,
                numasVal->Y[i],
                numasVal->subm[i]*handle->data_size
                );
    }
}