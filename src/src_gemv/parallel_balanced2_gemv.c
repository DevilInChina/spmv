//
// Created by kouushou on 2020/11/25.
//
#include "common_gemv.h"



void parallel_balanced2_gemv_Selected(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y,
        DOT_PRODUCT_WAY way
) {
    if(handle->status!=BALANCED2) {
        return;
    }
    int nthreads = handle->nthreads;
    int *Yid = handle->Yid;
    int *csrSplitter = handle->csrSplitter;
    int *Apinter = handle->Apinter;
    int *Start1 = handle->Start1;
    int *Start2 = handle->Start2;
    int *End2 = handle->End2;
    int *End1 = handle->End1;
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
    {
#pragma omp parallel default(shared)
        for (int tid = 0; tid < nthreads; tid++)
            Vector_Val_Y[Yid[tid]] = 0;
        int *Ysum = malloc(sizeof(int) * nthreads);
        int *Ypartialsum = malloc(sizeof(int) * nthreads);
#pragma omp parallel default(shared)
        for (int tid = 0; tid < nthreads; tid++) {
            if (Yid[tid] == -1) {
                for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                    Vector_Val_Y[u] =
                            dot_product(RowPtr[u+1]-RowPtr[u],
                                        ColIdx+RowPtr[u],
                                        Matrix_Val+RowPtr[u],Vector_Val_X);
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] > 1) {
                for (int u = Start1[tid]; u < End1[tid]; u++) {
                    Vector_Val_Y[u] =
                            dot_product(RowPtr[u+1]-RowPtr[u],
                                        ColIdx+RowPtr[u],
                                        Matrix_Val+RowPtr[u],Vector_Val_X);
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] <= 1) {
                Ysum[tid] = 0;
                Ypartialsum[tid] = dot_product(End2[tid]-Start2[tid],
                                               ColIdx+Start2[tid],Matrix_Val+Start2[tid],Vector_Val_X);
                Ysum[tid] += Ypartialsum[tid];
                Vector_Val_Y[Yid[tid]] += Ysum[tid];
            }
        }
        free(Ysum);
        free(Ypartialsum);
    }
}

void parallel_balanced2_gemv(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y){
    parallel_balanced2_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_NONE);
}


void parallel_balanced2_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y){
    parallel_balanced2_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_AVX2);
}


void parallel_balanced2_gemv_avx512(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y){
    parallel_balanced2_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_AVX512);
}