#include "inner_spmv.h"
#include <string.h>
#include <stdio.h>
typedef struct balancedYidEnv {
    BASIC_INT_TYPE *Brow, *beginIdx;/// nthread

    BASIC_INT_TYPE *Erow, *endIdx;/// nthread

    BASIC_INT_TYPE *splitter;/// 2*nthread
    BASIC_INT_TYPE *Type;/// nthread
    /// 1 for x-x ; both (mid could be zero)
    /// 0 for xxx ; single line

} balancedYidEnv, *balancedYidEnv_t;

void init_splitter_balancedYid(int nthreads, int nnzR,
                               int m, const BASIC_INT_TYPE *RowPtr,
                               BASIC_INT_TYPE *splitter,
                               BASIC_INT_TYPE *Type,
                               BASIC_INT_TYPE *Brow,
                               BASIC_INT_TYPE *Begin_Indx,
                               BASIC_INT_TYPE *Erow,
                               BASIC_INT_TYPE *End_Indx
) {
    int stride = (nnzR + nthreads - 1) / nthreads;
    for (int i = 1; i <= nthreads; ++i) {
        int begin_index = stride * (i - 1);
        int end_index = begin_index + stride;
        end_index = end_index > nnzR ? nnzR : end_index;

        int l = lower_bound(RowPtr, RowPtr + m + 1, begin_index);
        int r = lower_bound(RowPtr, RowPtr + m + 1, end_index);
        int type = -1;
        splitter[(i << 1) - 2] = l;
        splitter[(i << 1) - 1] = r - 1;

        Brow[i - 1] = l - 1;
        Erow[i - 1] = r - 1;
        Begin_Indx[i - 1] = begin_index;
        End_Indx[i - 1] = end_index;
        //printf("%d %d\n",Brow[i-1],Erow[i-1]);
        if (l == r) {/// a thread for single line;
            /// this thread will calculate from begin_index to end_index to Thread_begin_val
            Type[i - 1] = 0;
        } else {
            /// Thread_begin_val[i-1] = begin_index ~ RowPtr[Brow[i - 1] + 1] at Brow[i - 1]
            /// val from [l,r-1) calculate normally
            /// Thread_end_val[i-1] = RowPtr[Erow[i-1]] ~ end_index at Erow[i-1]
            Type[i - 1] = 1;

        }
    }
}

void parallel_balanced_Yid_get_handle(
        spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        BASIC_INT_TYPE nnzR
) {
    balancedYidEnv_t env = (balancedYidEnv_t) malloc(sizeof(balancedYidEnv));
    handle->extraHandle = env;
    env->splitter = (BASIC_INT_TYPE *) malloc(sizeof(BASIC_INT_TYPE) * 2 * handle->nthreads);
    env->Type = (BASIC_INT_TYPE *) malloc(sizeof(BASIC_INT_TYPE) * handle->nthreads);
    env->Brow = (BASIC_INT_TYPE *) malloc(sizeof(BASIC_INT_TYPE) * handle->nthreads);
    env->beginIdx = (BASIC_INT_TYPE *) malloc(sizeof(BASIC_INT_TYPE) * handle->nthreads);
    env->Erow = (BASIC_INT_TYPE *) malloc(sizeof(BASIC_INT_TYPE) * handle->nthreads);
    env->endIdx = (BASIC_INT_TYPE *) malloc(sizeof(BASIC_INT_TYPE) * handle->nthreads);

    init_splitter_balancedYid(handle->nthreads, nnzR, m, RowPtr,
                              env->splitter,
                              env->Type,
                              env->Brow,
                              env->beginIdx,
                              env->Erow,
                              env->endIdx
    );
}


void balancedYidHandleDestroy(spmv_Handle_t this_handle) {
    if (this_handle) {
        if (this_handle->extraHandle && this_handle->spmvMethod == Method_Balanced_Yid) {
            balancedYidEnv_t exh = (balancedYidEnv_t) this_handle->extraHandle;
            free(exh->splitter);
            free(exh->Type);
            free(exh->Brow);
            free(exh->Erow);
            free(exh->beginIdx);
            free(exh->endIdx);
            free(exh);
            this_handle->extraHandle = NULL;
        }
    }
}

void spmv_parallel_balanced_Yid_cpp_d(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const double *Matrix_Val,
        const double *Vector_Val_X,
        double *Vector_Val_Y
) {
    balancedYidEnv_t env = (balancedYidEnv_t) handle->extraHandle;
    BASIC_SIZE_TYPE nthreads = handle->nthreads;
    double *begin_val = (double *) malloc(sizeof(double) * nthreads);
    double *end_val = (double *) malloc(sizeof(double) * nthreads);
    memset(begin_val, 0, sizeof(double) * nthreads);
    memset(end_val, 0, sizeof(double) * nthreads);
    for (int i = 0; i < nthreads; ++i) {
        //printf("[%d] %d:%f %d:%f\n",env->Type[i],env->Brow[i],begin_val[i],env->Erow[i],end_val[i]);
        if(env->Brow[i]>0)
            Vector_Val_Y[env->Brow[i]] =0;
        Vector_Val_Y[env->Erow[i]] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < nthreads; ++i) {
        for (int line = env->splitter[i << 1]; line < env->splitter[i << 1 | 1]; ++line) {
            Dot_Product_Avx2_d(RowPtr[line + 1] - RowPtr[line],
                               ColIdx + RowPtr[line],
                               Matrix_Val + RowPtr[line],
                               Vector_Val_X,
                               Vector_Val_Y + line);
        }
        if (env->Type[i]) {
            Dot_Product_Avx2_d(RowPtr[env->Brow[i] + 1] - env->beginIdx[i],
                               ColIdx + env->beginIdx[i],
                               Matrix_Val + env->beginIdx[i],
                               Vector_Val_X,
                               begin_val + i
            );

            Dot_Product_Avx2_d(env->endIdx[i] - RowPtr[env->Erow[i]],
                               ColIdx + RowPtr[env->Erow[i]],
                               Matrix_Val + RowPtr[env->Erow[i]],
                               Vector_Val_X,
                               end_val + i
            );

        } else {
            Dot_Product_Avx2_d(env->endIdx[i] - env->beginIdx[i],
                               ColIdx + env->beginIdx[i],
                               Matrix_Val + env->beginIdx[i],
                               Vector_Val_X,
                               begin_val + i
            );
        }
    }
    for (int i = 0; i < nthreads; ++i) {
        //printf("[%d] %d:%f %d:%f\n",env->Type[i],env->Brow[i],begin_val[i],env->Erow[i],end_val[i]);
       if(env->Brow[i]>0)
            Vector_Val_Y[env->Brow[i]] += begin_val[i];
        Vector_Val_Y[env->Erow[i]] += end_val[i];
    }

    free(begin_val);
    free(end_val);
}

void spmv_parallel_balanced_Yid_cpp_s(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const float *Matrix_Val,
        const float *Vector_Val_X,
        float *Vector_Val_Y
) {
    balancedYidEnv_t env = (balancedYidEnv_t) handle->extraHandle;
    BASIC_SIZE_TYPE nthreads = handle->nthreads;
    float *begin_val = (float *) malloc(sizeof(float) * nthreads);
    float *end_val = (float *) malloc(sizeof(float) * nthreads);
    memset(begin_val, 0, sizeof(float) * nthreads);
    memset(end_val, 0, sizeof(float) * nthreads);
    for (int i = 0; i < nthreads; ++i) {
        //printf("[%d] %d:%f %d:%f\n",env->Type[i],env->Brow[i],begin_val[i],env->Erow[i],end_val[i]);
        if(env->Brow[i]>0)
            Vector_Val_Y[env->Brow[i]] =0;
        Vector_Val_Y[env->Erow[i]] = 0;
    }
#pragma omp parallel for
    for (int i = 0; i < nthreads; ++i) {
        for (int line = env->splitter[i << 1]; line < env->splitter[i << 1 | 1]; ++line) {
            Dot_Product_Avx2_s(RowPtr[line + 1] - RowPtr[line],
                               ColIdx + RowPtr[line],
                               Matrix_Val + RowPtr[line],
                               Vector_Val_X,
                               Vector_Val_Y + line);
        }
        if (env->Type[i]) {
            Dot_Product_Avx2_s(RowPtr[env->Brow[i] + 1] - env->beginIdx[i],
                               ColIdx + env->beginIdx[i],
                               Matrix_Val + env->beginIdx[i],
                               Vector_Val_X,
                               begin_val + i
            );

            Dot_Product_Avx2_s(env->endIdx[i] - RowPtr[env->Erow[i]],
                               ColIdx + RowPtr[env->Erow[i]],
                               Matrix_Val + RowPtr[env->Erow[i]],
                               Vector_Val_X,
                               end_val + i
            );

        } else {
            Dot_Product_Avx2_s(env->endIdx[i] - env->beginIdx[i],
                               ColIdx + env->beginIdx[i],
                               Matrix_Val + env->beginIdx[i],
                               Vector_Val_X,
                               begin_val + i
            );
        }
    }
    for (int i = 0; i < nthreads; ++i) {
        //printf("[%d] %d:%f %d:%f\n",env->Type[i],env->Brow[i],begin_val[i],env->Erow[i],end_val[i]);
        if(env->Brow[i]>0)
            Vector_Val_Y[env->Brow[i]] += begin_val[i];
        Vector_Val_Y[env->Erow[i]] += end_val[i];
    }

    free(begin_val);
    free(end_val);
}




void spmv_parallel_balancedYid_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y
) {
    if (handle->data_size == sizeof(double)) {
        spmv_parallel_balanced_Yid_cpp_d(handle, m,
                                     RowPtr, ColIdx,
                                     (double *) Matrix_Val,
                                     (double *) Vector_Val_X,
                                     (double *) Vector_Val_Y);
    } else {
        spmv_parallel_balanced_Yid_cpp_s(handle, m, RowPtr, ColIdx,
                                     (float *) Matrix_Val,
                                     (float *) Vector_Val_X,
                                     (float *) Vector_Val_Y);
    }
}
