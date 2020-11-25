//
// Created by kouushou on 2020/11/25.
//

#include "common_gemv.h"
void parallel_balanced_gemv(
                            const gemv_Handle_t handle,
                            GEMV_INT_TYPE m,
                            const GEMV_INT_TYPE* RowPtr,
                            const GEMV_INT_TYPE* ColIdx,
                            const GEMV_VAL_TYPE* Matrix_Val,
                            const GEMV_VAL_TYPE* Vector_Val_X,
                            GEMV_VAL_TYPE*       Vector_Val_Y) {
    const int *csrSplitter = handle->csrSplitter;
    const int nthreads = handle->nthreads;
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++) {
        for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
            float sum = 0;
            for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) {
                sum += Matrix_Val[j] * Vector_Val_X[ColIdx[j]];
            }
            Vector_Val_Y[u] = sum;
        }
    }
}

void parallel_balanced_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y) {
    const int *csrSplitter = handle->csrSplitter;
    const int nthreads = handle->nthreads;

#pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++)
    {
        for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
        {
            __m256 res = _mm256_setzero_ps();
            float sum = 0;
            int dif = RowPtr[u+1] - RowPtr[u];
            int nloop = dif / 8;
            int remainder = dif % 8;
            for (int li = 0; li < nloop; li++)
            {
                int j = RowPtr[u] + li * 8;
                __m256 vecv = _mm256_loadu_ps(&Matrix_Val[j]);
                __m256i veci =  _mm256_loadu_si256((__m256i *)(&ColIdx[j]));
                __m256 vecx = _mm256_i32gather_ps(Vector_Val_X, veci, 4);
                res = _mm256_fmadd_ps(vecv, vecx, res);
            }
            //Y[u] += _mm256_reduce_add_ps(res);
            sum += hsum_avx(res);

            for (int j = RowPtr[u] + nloop * 8; j < RowPtr[u + 1]; j++) {
                sum += Matrix_Val[j] * Vector_Val_X[ColIdx[j]];
            }
            Vector_Val_Y[u] = sum;
        }

    }
}


