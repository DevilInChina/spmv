//
// Created by kouushou on 2020/11/25.
//
#include <gemv.h>

void parallel_balanced2_gemv(
        GEMV_INT_TYPE        nthreads,
        const GEMV_INT_TYPE* Yid,
        const GEMV_INT_TYPE* Apinter,
        const GEMV_INT_TYPE* Start1,
        const GEMV_INT_TYPE* End1,
        const GEMV_INT_TYPE* Start2,
        const GEMV_INT_TYPE* End2,
        const GEMV_INT_TYPE* csrSplitter,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y) {

#pragma omp parallel default(shared)
    for (int tid = 0; tid < nthreads; tid++)
        Vector_Val_Y[Yid[tid]] = 0;
    int *Ysum = malloc(sizeof(int)*nthreads);
    int *Ypartialsum = malloc(sizeof(int)*nthreads);
#pragma omp parallel default(shared)
    for (int tid = 0; tid < nthreads; tid++)
    {
        if (Yid[tid] == -1)
        {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
            {
                float sum = 0;
                for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++)
                {
                    sum += Matrix_Val[j] * Vector_Val_X[ColIdx[j]];
                }
                Vector_Val_Y[u] = sum;
            }
        }
        if (Yid[tid] != -1 && Apinter[tid] > 1)
        {
            for (int u = Start1[tid]; u < End1[tid]; u++)
            {
                float sum = 0;
                for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++)
                {
                    sum += Matrix_Val[j] * Vector_Val_X[ColIdx[j]];
                }
                Vector_Val_Y[u] = sum;
            }
        }
        if (Yid[tid] != -1 && Apinter[tid] <= 1)
        {
            Ysum[tid] = 0;
            Ypartialsum[tid] = 0;
            for (int j = Start2[tid]; j < End2[tid]; j++)
            {
                Ypartialsum[tid] += Matrix_Val[j] * Vector_Val_X[ColIdx[j]];
            }
            Ysum[tid] += Ypartialsum[tid];
            Vector_Val_Y[Yid[tid]] += Ysum[tid];
        }
    }
    free(Ysum);
    free(Ypartialsum);
}