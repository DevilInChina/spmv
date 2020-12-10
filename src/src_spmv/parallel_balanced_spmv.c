//
// Created by kouushou on 2020/11/25.
//

#include <inner_spmv.h>



void spmv_parallel_balanced_Selected(
        const gemv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y
) {
    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY way = handle->vectorizedWay;
    dot_product_function dotProductFunction = inner_basic_GetDotProduct(size);

    const int *csrSplitter = handle->csrSplitter;
    const BASIC_SIZE_TYPE nthreads = handle->nthreads;
    {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                dotProductFunction(RowPtr[u + 1] - RowPtr[u], ColIdx + RowPtr[u],
                                     Matrix_Val + RowPtr[u]*size, Vector_Val_X,Vector_Val_Y+u*size,way);
            }
        }
    }
}



