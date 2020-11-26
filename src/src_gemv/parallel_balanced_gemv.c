//
// Created by kouushou on 2020/11/25.
//

#include "common_gemv.h"
void parallel_balanced_gemv_Selected(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y,
        DOT_PRODUCT_WAY way
        ) {
    if(handle->status!=BALANCED) {
        return;
    }
    float (*dot_product)(GEMV_INT_TYPE len, const GEMV_INT_TYPE *indx, const float *Val, const float *X);
    switch (way) {
        case DOT_AVX2: {
            dot_product = gemv_s_dotProduct_avx2;
        }
            break;
        case DOT_AVX512: {
            dot_product = gemv_s_dotProduct_avx512;
        }
            break;
        default: {
            dot_product = gemv_s_dotProduct;
        }
            break;
    }

    const int *csrSplitter = handle->csrSplitter;
    const int nthreads = handle->nthreads;
    {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                Vector_Val_Y[u] = dot_product(
                        RowPtr[u + 1] - RowPtr[u], ColIdx + RowPtr[u],
                        Matrix_Val + RowPtr[u], Vector_Val_X);
            }
        }
    }
}
void parallel_balanced_gemv(
                            const gemv_Handle_t handle,
                            GEMV_INT_TYPE m,
                            const GEMV_INT_TYPE* RowPtr,
                            const GEMV_INT_TYPE* ColIdx,
                            const GEMV_VAL_TYPE* Matrix_Val,
                            const GEMV_VAL_TYPE* Vector_Val_X,
                            GEMV_VAL_TYPE*       Vector_Val_Y) {
    parallel_balanced_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_NONE);
}

void parallel_balanced_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y) {
    parallel_balanced_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_AVX2);
}


void parallel_balanced_gemv_avx512(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y) {
    parallel_balanced_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_AVX512);
}