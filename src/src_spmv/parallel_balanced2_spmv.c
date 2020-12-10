//
// Created by kouushou on 2020/11/25.
//
#include <inner_spmv.h>


void spmv_parallel_balanced2_Selected(
        const gemv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y
) {
    int nthreads = handle->nthreads;
    int *Yid = handle->Yid;
    int *csrSplitter = handle->csrSplitter;
    int *Apinter = handle->Apinter;
    int *Start1 = handle->Start1;
    int *Start2 = handle->Start2;
    int *End2 = handle->End2;
    int *End1 = handle->End1;
    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY way = handle->vectorizedWay;
    dot_product_function dotProductFunction = inner_basic_GetDotProduct(size);

    for (int tid = 0; tid < nthreads; tid++) {
        if(Yid[tid]!=-1) {
            CONVERT_EQU(Vector_Val_Y+Yid[tid],size,0);
        }
    }

    void *Ysum = malloc(size * nthreads);
    void *Ypartialsum = malloc(size * nthreads);
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++) {
        if (Yid[tid] == -1) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                dotProductFunction(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u]*size, Vector_Val_X,Vector_Val_Y+u*size,way);
            }
        }
        if (Yid[tid] != -1 && Apinter[tid] > 1) {/// not in usage
            for (int u = Start1[tid]; u < End1[tid]; u++) {
                dotProductFunction(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u]*size, Vector_Val_X,Vector_Val_Y+u*size,way);
            }
        }
        if (Yid[tid] != -1 && Apinter[tid] <= 1) {
            CONVERT_EQU(Ysum+tid*size,size,0);
            dotProductFunction(End2[tid] - Start2[tid],
                               ColIdx + Start2[tid], Matrix_Val + Start2[tid]*size, Vector_Val_X,
                               Ypartialsum+tid*size,way
                               );

            CONVERT_ADDEQU(Ysum+tid*size,size,Ypartialsum+tid*size);

            CONVERT_ADDEQU(Vector_Val_Y+Yid[tid]*size,size,Ysum+tid*size);
        }
    }
    free(Ysum);
    free(Ypartialsum);
}


