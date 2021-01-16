//
// Created by kouushou on 2021/1/9.
//
#include "inner_spmv.h"

#include <numa.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>


typedef struct numa_spmv_parameter {
    int nthreads, numanodes, m, coreidx, alloc;
    spmv_Handle_t handle;
} numa_spmv_parameter, *numa_spmv_parameter_t, numaspmv;

typedef struct NumaEnvironment {
    int **subrowptrA, **subcolidxA;
    double **subvalA, **X, **Y;
    numa_spmv_parameter_t p;
    int numanodes;
    int *subm, *subm_ex, *subX_ex, *subX, *subnnz, *subnnz_ex;
    int PARTS;
} NumaEnvironment, *NumaEnvironment_t;


void inner_exclusive_scan(BASIC_INT_TYPE *input, int length) {
    if (length == 0 || length == 1)
        return;

    BASIC_INT_TYPE old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++) {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void numaHandleDestory(spmv_Handle_t handle) {
    if (handle) {
        if (handle->extraHandle && handle->spmvMethod == Method_Numa) {
            NumaEnvironment_t numasVal = handle->extraHandle;
            for (int i = 0; i < numasVal->numanodes; ++i) {

                numa_free(numasVal->subrowptrA[i],
                          sizeof(int) * (numasVal->subm[numasVal->p[i].alloc] + 1));

                numa_free(numasVal->subcolidxA[i],
                          sizeof(int) * numasVal->subnnz[numasVal->p[i].alloc]);

                numa_free(numasVal->subvalA[i],
                          handle->data_size * numasVal->subnnz[numasVal->p[i].alloc]);

                numa_free(numasVal->X[i],
                          handle->data_size * numasVal->subX[numasVal->p[i].alloc]);

                numa_free(numasVal->Y[i],
                          handle->data_size * numasVal->subX[numasVal->p[i].alloc]);

            }

            free(numasVal->p);
            free(numasVal->subX);
            free(numasVal->subX_ex);
            free(numasVal->subm);
            free(numasVal->subm_ex);
            free(numasVal->subnnz);
            free(numasVal->subnnz_ex);
            free(handle->extraHandle);
            handle->extraHandle = NULL;
        }
    }
}


void *spmv_numa(void *arg) {

    numaspmv *pn = (numaspmv *) arg;
    NumaEnvironment_t numaEnvironment = pn->handle->extraHandle;
    int me = pn->alloc;
    numa_run_on_node(me);
    int m = pn->m;
    int nthreads = pn->nthreads;
    int numanodes = pn->numanodes;
    int coreidx = pn->coreidx;
    int eachnumathreads = nthreads / numanodes;
    int task = ceil((double) m / (double) eachnumathreads);
    int start = coreidx * task;
    int end = (coreidx + 1) * task > m ? m : (coreidx + 1) * task;
    //printf("numanode %d, coreindex %d, m %d, nthreads %d, eachnumathreads %d, start %d, end %d\n",pn->alloc,coreidx,pn->m,nthreads,eachnumathreads,start,end);
    double *val = numaEnvironment->subvalA[me];
    //VALUE_TYPE *x = X[me];
    double *y = numaEnvironment->Y[me];
    int *rpt = numaEnvironment->subrowptrA[me];
    int *col = numaEnvironment->subcolidxA[me];
    for (int u = start; u < end; u++) {
        double sum = 0;
        for (int j = rpt[u]; j < rpt[u + 1]; j++) {
            int Xpos = col[j] / numaEnvironment->subX[0];
            int remainder = col[j] - numaEnvironment->subX_ex[Xpos];
            sum += val[j] * numaEnvironment->X[Xpos][remainder];
            //
        }
        printf("%f\n",sum);
        y[u] = sum;
        //if(me==7)
        //printf("y[%d][%d]%.2f\n",me,u,sum);
    }

}

int numa_spmv_get_handle_Selected(spmv_Handle_t handle,
                                  int PARTS,
                                  BASIC_INT_TYPE m, BASIC_INT_TYPE n,
                                  const BASIC_INT_TYPE *RowPtr,
                                  const BASIC_INT_TYPE *ColIdx,
                                  const void *Matrix_Val
) {
    int numanodes = numa_max_node() + 1;
    //if(numanodes==1)return 0;
    handle->extraHandle = malloc(sizeof(NumaEnvironment));
    NumaEnvironment_t numaVal = handle->extraHandle;

    const BASIC_INT_TYPE *rowptrA = RowPtr;
    const BASIC_INT_TYPE *colidxA = ColIdx;
    int nthreads = handle->nthreads;

    int **subrowptrA;
    int **subcolidxA;
    const double *valA = Matrix_Val;
    double **subvalA;
    double **X;
    double **Y;

    int eachnumacores = nthreads / numanodes;

    int *subrowpos = (int *) malloc(sizeof(int) * (PARTS + 1));
    int *subm = (int *) malloc(sizeof(int) * PARTS);
    int *subm_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
    int *subX = (int *) malloc(sizeof(int) * PARTS);
    int *subX_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
    int *subnnz = (int *) malloc(sizeof(int) * PARTS);
    int *subnnz_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
    int i, j, k;
    for (i = 0; i < PARTS; i++)
        subrowpos[i] = (ceil((double) m / (double) PARTS)) * i > m ? m : ((ceil((double) m / (double) PARTS)) * i);
    subrowpos[PARTS] = m;
    for (i = 0; i < (PARTS - 1); i++) {
        subX[i] = ceil((double) n / (double) PARTS);
    }
    subX[PARTS - 1] = n - subX[0] * (PARTS - 1);
    for (i = 0; i < PARTS; i++) {
        subX_ex[i] = subX[i];
    }
    inner_exclusive_scan(subX_ex, PARTS + 1);
    for (i = 0; i < PARTS; i++) {
        subm[i] = subrowpos[i + 1] - subrowpos[i];
        subm_ex[i] = subrowpos[i + 1] - subrowpos[i];
    }
    inner_exclusive_scan(subm_ex, PARTS + 1);
    for (i = 0; i < PARTS; i++) {
        subnnz[i] = rowptrA[subrowpos[i] + subm[i]] - rowptrA[subrowpos[i]];
        subnnz_ex[i] = subnnz[i];
    }
    inner_exclusive_scan(subnnz_ex, PARTS + 1);
    numaspmv *p = (numaspmv *) malloc(nthreads * sizeof(numaspmv));
    for (i = 0; i < nthreads; i++) {
        p[i].alloc = i % numanodes;
        p[i].numanodes = numanodes;
        p[i].nthreads = nthreads;
        p[i].handle = handle;
    }
    for (i = 0; i < eachnumacores; i++) {
        for (j = 0; j < numanodes; j++) {
            p[i * numanodes + j].coreidx = i;
            p[i * numanodes + j].m = subm[j];
        }
    }
    subrowptrA = (int **) malloc(sizeof(int *) * nthreads);
    subcolidxA = (int **) malloc(sizeof(int *) * nthreads);
    subvalA = malloc(sizeof(void *) * nthreads);
    X = malloc(sizeof(void *) * nthreads);
    Y = malloc(sizeof(void *) * nthreads);

    for (i = 0; i < nthreads; i++) {
        subrowptrA[i] = numa_alloc_onnode(sizeof(int) * (subm[p[i].alloc] + 1), p[i].alloc);
        subcolidxA[i] = numa_alloc_onnode(sizeof(int) * subnnz[p[i].alloc], p[i].alloc);
        subvalA[i] = numa_alloc_onnode(handle->data_size * subnnz[p[i].alloc], p[i].alloc);
        X[i] = numa_alloc_onnode(handle->data_size * subX[p[i].alloc], p[i].alloc);
        Y[i] = numa_alloc_onnode(handle->data_size * subm[p[i].alloc], p[i].alloc);
    }
    int currentcore = 0;
    for (i = 0; i < numanodes; i++) {
        for (j = 0; j < eachnumacores; j++) {
            for (k = 0; k <= subm[i]; k++) {
                currentcore = i + j * eachnumacores;
                if (currentcore < nthreads) {
                    subrowptrA[currentcore][k] = rowptrA[subrowpos[i] + k];
                }
            }
        }
        for (j = 0; j < eachnumacores; j++) {
            for (k = 0; k < subnnz[i]; k++) {
                currentcore = i + j * eachnumacores;
                if (currentcore < nthreads) {
                    subcolidxA[currentcore][k] = colidxA[subnnz_ex[i] + k];
                    subvalA[currentcore][k] = valA[subnnz_ex[i] + k];
                }
            }
        }

    }
    for (i = 0; i < nthreads; i++) {
        if (i % numanodes != 0) {
            int temprpt = subrowptrA[i][0];
            for (j = 0; j <= subm[i % numanodes]; j++) {
                subrowptrA[i][j] -= temprpt;
            }
        }
    }
    numaVal->X = X;
    numaVal->Y = Y;
    numaVal->numanodes = numanodes;
    numaVal->subvalA = subvalA;
    numaVal->subcolidxA = subcolidxA;
    numaVal->subrowptrA = subrowptrA;
    numaVal->p = p;
    numaVal->PARTS = PARTS;
    numaVal->subnnz_ex = subnnz_ex;
    numaVal->subX_ex = subX_ex;
    numaVal->subm_ex = subm_ex;
    numaVal->subnnz = subnnz;
    numaVal->subX = subX;
    numaVal->subm = subm;

    return numanodes;
}


void spmv_numa_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y
) {
    NumaEnvironment_t numasVal = handle->extraHandle;
    int numanodes = numasVal->numanodes;
    int eachnumacores = handle->nthreads / numasVal->numanodes;
    memset(Vector_Val_Y,0,sizeof(handle->data_size)*m);
    for (int i = 0; i < numanodes; ++i) {
        for (int j = 0; j < eachnumacores; j++) {
            for (int k = 0; k < numasVal->subX[i]; k++) {
                int currentcore = i + j * eachnumacores;
                if (currentcore < handle->nthreads) {
                    numasVal->X[currentcore][k] = ((double *)Matrix_Val)[numasVal->subX_ex[i] + k];
                }
            }
        }
    }
    pthread_t *threads = (pthread_t *) malloc(handle->nthreads * sizeof(pthread_t));
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);

    for (int i = 0; i < handle->nthreads; i++) {

        pthread_create(threads + i, &pthread_custom_attr, spmv_numa,
                       (void *) (numasVal->p + i));
    }
    for (int i = 0; i < handle->nthreads; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < numasVal->PARTS; i++) {
        for (int j = 0; j < numasVal->subm[i]; j++) {
            ((double*)Vector_Val_Y) [numasVal->subm_ex[i] + j] = numasVal->Y[i][j];
        }

    }

    free(threads);
    pthread_attr_destroy(&pthread_custom_attr);
}