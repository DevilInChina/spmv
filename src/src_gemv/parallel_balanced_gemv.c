//
// Created by kouushou on 2020/11/25.
//

#include <gemv.h>
void parallel_balanced_gemv(
                            GEMV_INT_TYPE        nthreads,
                            const GEMV_INT_TYPE* csrSplitter,
                            GEMV_INT_TYPE m,
                            const GEMV_INT_TYPE* RowPtr,
                            const GEMV_INT_TYPE* ColIdx,
                            const GEMV_VAL_TYPE* Matrix_Val,
                            const GEMV_VAL_TYPE* Vector_Val_X,
                            GEMV_VAL_TYPE*       Vector_Val_Y) {
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

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}


void parallel_balanced_get_csrSplitter(
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE*RowPtr,
        GEMV_INT_TYPE nnzR,
        GEMV_INT_TYPE nthreads,
        GEMV_INT_TYPE**csrSplitter) {
    *csrSplitter = (int *) malloc((nthreads + 1) * sizeof(int));
    //int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));
    int stridennz = ceil((double) nnzR / (double) nthreads);

    for (int i = 0; i < 2; ++i);
#pragma omp parallel default(none) shared(nthreads, stridennz, nnzR, RowPtr, csrSplitter, m)
    for (int tid = 0; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        *csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
    }
}