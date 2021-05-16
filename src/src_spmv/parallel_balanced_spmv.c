//
// Created by kouushou on 2020/11/25.
//

#include "inner_spmv.h"

void balancedHandleDestroy(spmv_Handle_t this_handle) {
    if (this_handle) {
        if (this_handle->extraHandle && this_handle->spmvMethod == Method_Balanced) {
            free(this_handle->extraHandle);
            this_handle->extraHandle = NULL;
        }
    }

}

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int key_input,
                                        const int size) {
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start) {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

void init_csrSplitter_balanced(int nthreads, int nnzR,
                               int m, const BASIC_INT_TYPE *RowPtr, BASIC_INT_TYPE *csrSplitter) {
    int stridennz = (nnzR + nthreads - 1) / nthreads;

    csrSplitter[0] = 0;
    for (int tid = 1; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        int spl = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
        if (spl == csrSplitter[tid - 1]) {
            spl = m > (spl + 1) ? (spl + 1) : m;
            csrSplitter[tid] = spl;
        } else {
            csrSplitter[tid] = spl;
        }
    }
}

void parallel_balanced_get_handle(
        spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        BASIC_INT_TYPE nnzR
) {
    BASIC_SIZE_TYPE nthreads = handle->nthreads;

    int *csrSplitter = (int *) malloc((nthreads + 1) * sizeof(int));
    //int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));
    init_csrSplitter_balanced((int) nthreads, nnzR, m, RowPtr, csrSplitter);

    (handle)->extraHandle = csrSplitter;

}


 void spmv_parallel_balanced_cpp_d(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const double *Matrix_Val,
        const double *Vector_Val_X,
        double *Vector_Val_Y
) {
    const int *csrSplitter = (int *) handle->extraHandle;
    const BASIC_SIZE_TYPE nthreads = handle->nthreads;
    {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                Dot_Product_Avx2_d(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u],
                                   Vector_Val_X,
                                   Vector_Val_Y + u);
            }
        }
    }
}

 void spmv_parallel_balanced_cpp_s(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const float *Matrix_Val,
        const float *Vector_Val_X,
        float *Vector_Val_Y
) {
    const int *csrSplitter = (int *) handle->extraHandle;
    const BASIC_SIZE_TYPE nthreads = handle->nthreads;
    {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                Dot_Product_Avx2_s(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u],
                                   Vector_Val_X,
                                   Vector_Val_Y + u);
            }
        }
    }
}

void spmv_parallel_balanced_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y
) {
    if (handle->data_size == sizeof(double)) {
        spmv_parallel_balanced_cpp_d(handle, m,
                                   RowPtr, ColIdx,
                                   (double *) Matrix_Val,
                                   (double *) Vector_Val_X,
                                   (double *) Vector_Val_Y);
    } else {
        spmv_parallel_balanced_cpp_s(handle, m, RowPtr, ColIdx,
                                   (float *) Matrix_Val,
                                   (float *) Vector_Val_X,
                                   (float *) Vector_Val_Y);
    }
}


