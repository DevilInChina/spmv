//
// Created by kouushou on 2021/1/9.
//
#ifdef NUMA
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
    void **subvalA, **X, **Y;
    numa_spmv_parameter_t p;
    int numanodes;
    int *subm, *subm_ex, *subX_ex, *subX, *subnnz, *subnnz_ex;
    int PARTS;
    int *RowPtr;
    int *ColIdx;
    void *Val;
} NumaEnvironment, *NumaEnvironment_t;



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
            free(numasVal->RowPtr);
            free(numasVal->ColIdx);
            free(numasVal->Val);
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
    void *val = numaEnvironment->subvalA[me];
    //VALUE_TYPE *x = X[me];
    void *y = numaEnvironment->Y[me];
    int *rpt = numaEnvironment->subrowptrA[me];
    int *col = numaEnvironment->subcolidxA[me];
    if(pn->handle->data_size==sizeof(double )) {

        for (int u = start; u < end; u++) {
            double sum = 0;
            for (int j = rpt[u]; j < rpt[u + 1]; j++) {
                int Xpos = col[j] / numaEnvironment->subX[0];
                int remainder = col[j] - numaEnvironment->subX_ex[Xpos];
                sum += ((double *)val)[j] * ((double **)numaEnvironment->X)[Xpos][remainder];
                //
            }
            ((double *)y)[u] = sum;
            //if(me==7)
            //printf("y[%d][%d]%.2f\n",me,u,sum);
        }
    }else{
        for (int u = start; u < end; u++) {
            float sum = 0;
            for (int j = rpt[u]; j < rpt[u + 1]; j++) {
                int Xpos = col[j] / numaEnvironment->subX[0];
                int remainder = col[j] - numaEnvironment->subX_ex[Xpos];
                sum += ((float *)val)[j] * ((float **)numaEnvironment->X)[Xpos][remainder];
                //
            }
            ((float *)y)[u] = sum;
            //if(me==7)
            //printf("y[%d][%d]%.2f\n",me,u,sum);
        }
    }

}
void partition(const int m,const int n,const int PARTS,
               const int nthreads,const int numanodes,
               const int *rowptrA,const int *colidxA,
               const void*valA,numa_spmv_parameter_t p,BASIC_SIZE_TYPE type_size,
               int **subrowptrA,int **subcolidxA,void **X,void **Y,void **subvalA,
               int *subm,
               int *subm_ex,
               int *subX,
               int *subX_ex,
               int *subnnz,
               int *subnnz_ex
               ){
    int eachnumacores = nthreads / numanodes;
    int i,j,k;

    int *subrowpos = (int *) malloc(sizeof(int) * (PARTS + 1));

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
    for (i = 0; i < nthreads; i++) {
        p[i].alloc = i % numanodes;
        p[i].numanodes = numanodes;
        p[i].nthreads = nthreads;
    }
    for (i = 0; i < eachnumacores; i++) {
        for (j = 0; j < numanodes; j++) {
            p[i * numanodes + j].coreidx = i;
            p[i * numanodes + j].m = subm[j];
        }
    }


    for (i = 0; i < nthreads; i++) {
        subrowptrA[i] = numa_alloc_onnode(sizeof(int) * (subm[p[i].alloc] + 1), p[i].alloc);
        subcolidxA[i] = numa_alloc_onnode(sizeof(int) * subnnz[p[i].alloc], p[i].alloc);
        subvalA[i] = numa_alloc_onnode(type_size * subnnz[p[i].alloc], p[i].alloc);
        X[i] = numa_alloc_onnode(type_size * subX[p[i].alloc], p[i].alloc);
        Y[i] = numa_alloc_onnode(type_size * subm[p[i].alloc], p[i].alloc);
    }
    int currentcore = 0;
    for (i = 0; i < numanodes; i++) {

        for (j = 0; j < eachnumacores; j++) {
            currentcore = i + j * eachnumacores;
            if (currentcore < nthreads) {
                memcpy(subrowptrA[currentcore],rowptrA+subrowpos[i],(1+subm[i])*sizeof(int));
                memcpy(subcolidxA[currentcore], colidxA + subnnz_ex[i], subnnz[i] * sizeof(int));
                memcpy(subvalA[currentcore], valA + subnnz_ex[i] * type_size, subnnz[i] * type_size);
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
    free(subrowpos);
}
int numa_spmv_get_handle_Selected(spmv_Handle_t handle,
                                  BASIC_INT_TYPE m, BASIC_INT_TYPE n,
                                  const BASIC_INT_TYPE *RowPtr,
                                  const BASIC_INT_TYPE *ColIdx,
                                  const void *Matrix_Val
) {
    int numanodes = numa_max_node() + 1;
    int PARTS = numanodes;
    //if(numanodes==1)return 0;
    handle->extraHandle = malloc(sizeof(NumaEnvironment));
    NumaEnvironment_t numaVal = handle->extraHandle;


    int nthreads = handle->nthreads;
    int nnz = RowPtr[m] - RowPtr[0];

    int *RowPtrCpy = malloc(sizeof(int )*(m+1));
    int *ColIdxCpy = malloc(sizeof(int )*(nnz));
    void *ValCpy = malloc(handle->data_size*nnz);

    memcpy(RowPtrCpy,RowPtr,sizeof(int )*(m+1));
    memcpy(ColIdxCpy,ColIdx,sizeof(int )*(nnz));
    memcpy(ValCpy,Matrix_Val,handle->data_size*(nnz));


    //metis_partitioning(n, m, nnz, PARTS, RowPtrCpy, ColIdxCpy, ValCpy,handle->data_size);



    int **subrowptrA;
    int **subcolidxA;
    void **subvalA;
    void **X;
    void **Y;
    subrowptrA = (int **) malloc(sizeof(int *) * nthreads);
    subcolidxA = (int **) malloc(sizeof(int *) * nthreads);
    subvalA =  malloc(sizeof(void *) * nthreads);
    X =  malloc(sizeof(void *) * nthreads);
    Y =  malloc(sizeof(void *) * nthreads);
    int eachnumacores = nthreads / numanodes;


    int *subm = (int *) malloc(sizeof(int) * PARTS);
    int *subm_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
    int *subX = (int *) malloc(sizeof(int) * PARTS);
    int *subX_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
    int *subnnz = (int *) malloc(sizeof(int) * PARTS);
    int *subnnz_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
    numaspmv *p = (numaspmv *) malloc(nthreads * sizeof(numaspmv));


    partition(m,n,PARTS,nthreads,numanodes,RowPtrCpy,ColIdxCpy,ValCpy,p,handle->data_size,
              subrowptrA,subcolidxA,X,Y,subvalA,subm,subm_ex,subX,subX_ex,subnnz,subnnz_ex
    );

    for(int i = 0 ; i < nthreads ; ++i){
        p[i].handle = handle;
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
    numaVal->subnnz = subnnz;
    numaVal->subX_ex = subX_ex;
    numaVal->subX = subX;
    numaVal->subm_ex = subm_ex;
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
    memset(Vector_Val_Y, 0, handle->data_size * m);

    for (int i = 0; i < numanodes; ++i) {
        for (int j = 0; j < eachnumacores; j++) {
            int currentcore = i + j * eachnumacores;
            if(currentcore<handle->nthreads){
                memcpy(numasVal->X[currentcore],
                       Vector_Val_X+numasVal->subX_ex[i]*handle->data_size,
                       numasVal->subX[i]*handle->data_size
                );
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
    for (int i = 0; i < numanodes; i++) {
        memcpy(Vector_Val_Y+numasVal-> subm_ex[i]*handle->data_size,
               numasVal->Y[i],
               numasVal-> subm[i]*handle->data_size);
    }

    free(threads);
    pthread_attr_destroy(&pthread_custom_attr);
}
#endif