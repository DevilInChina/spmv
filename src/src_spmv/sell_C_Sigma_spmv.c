//
// Created by kouushou on 2020/12/4.
//
#include "inner_spmv.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX(a, b) ((a)>(b)?(a):(b))
typedef struct Row_Block {
    const BASIC_INT_TYPE *indxBegin;
    const void *valBegin;
    BASIC_INT_TYPE length;
    BASIC_INT_TYPE rowNumber;
} Row_Block, *Row_Block_t;


typedef struct Sigma_Block {
    BASIC_INT_TYPE C;
    BASIC_INT_TYPE times;
    BASIC_INT_TYPE *ld;
    BASIC_INT_TYPE *ColIndex;
    BASIC_INT_TYPE *RowIndex;
    BASIC_INT_TYPE total;
    void *ValT;
    //void *Y;
} Sigma_Block, *Sigma_Block_t;

void C_Block_destory(Sigma_Block_t this_block) {
    free(this_block->RowIndex);
    free(this_block->ColIndex);
    free(this_block->ValT);
    //free(this_block->Y);
    free(this_block->ld);
}

typedef struct sigmaEnv {
    BASIC_INT_TYPE Sigma, C;
    BASIC_INT_TYPE banner;
    Sigma_Block_t sigmaBlock;
} sigmaEnv, *sigmaEnv_t;

void sellCSigmaHandleDestroy(spmv_Handle_t this_handle) {
    if (this_handle && this_handle->spmvMethod == Method_SellCSigma) {

        if (this_handle->extraHandle) {
            sigmaEnv_t sigenv = (sigmaEnv_t) this_handle->extraHandle;
            if(sigenv->banner) {
                int siz = sigenv->banner / sigenv->Sigma;
                for (int i = 0; i < siz; ++i) {
                    C_Block_destory(sigenv->sigmaBlock + i);
                }
            }
            free(this_handle->extraHandle);
        }
    }
}

void spmv_Sigma_Blocks_init(Sigma_Block_t SigmaBeginner, int C, int Sigma,
                            const BASIC_INT_TYPE *RowPtr,
                            Row_Block_t *rowBlock,
                            BASIC_SIZE_TYPE size) {
    int times = Sigma / C;
    SigmaBeginner->times = times;
    SigmaBeginner->C = C;
    SigmaBeginner->ld = (int *) malloc(sizeof(BASIC_INT_TYPE) * (times + 1));
    int *loc = (int *) malloc(sizeof(BASIC_INT_TYPE) * times);
    SigmaBeginner->ld[0] = 0;
    for (int i = 0; i < times; ++i) {
        int maxs = 0;
        loc[i] = 0;
        for (int j = 0; j < C; ++j) {
            int cld = RowPtr[rowBlock[C * i + j]->rowNumber + 1] - RowPtr[rowBlock[C * i + j]->rowNumber];
            if (cld > maxs) {
                loc[i] = j;
                maxs = cld;
            }
        }
        SigmaBeginner->ld[i + 1] = maxs + SigmaBeginner->ld[i];
    }
    if (SigmaBeginner->ld[times] == 0) {
        free(SigmaBeginner->ld);
        SigmaBeginner->ld = NULL;
        SigmaBeginner->ColIndex = NULL;
        SigmaBeginner->RowIndex = NULL;
        SigmaBeginner->ValT = NULL;
        //SigmaBeginner->Y = NULL;
        return;
    }
    SigmaBeginner->ColIndex = (int *) aligned_alloc(ALIGENED_SIZE,
                                                    sizeof(BASIC_INT_TYPE) * C * SigmaBeginner->ld[times]);
    SigmaBeginner->RowIndex = (int *) aligned_alloc(ALIGENED_SIZE, sizeof(BASIC_INT_TYPE) * C * times);
    SigmaBeginner->ValT = aligned_alloc(ALIGENED_SIZE, size * C * SigmaBeginner->ld[times]);
    memset(SigmaBeginner->ValT, 0, size * C * SigmaBeginner->ld[times]);
    memset(SigmaBeginner->ColIndex, 0, sizeof(BASIC_INT_TYPE) * C * SigmaBeginner->ld[times]);
    //SigmaBeginner->Y = aligned_alloc(ALIGENED_SIZE, size * C * times);
    for (int i = 0; i < Sigma; ++i) {
        SigmaBeginner->RowIndex[i] = rowBlock[i]->rowNumber;
    }
    SigmaBeginner->total = SigmaBeginner->ld[times];
    //*total += C * SigmaBeginner->ld[times];
    for (int k = 0; k < times; ++k) {/// ini C
        for (int i = 0; i < C; ++i) {
            int j = 0;
            for (; j < RowPtr[rowBlock[i + k * C]->rowNumber + 1] - RowPtr[rowBlock[i + k * C]->rowNumber]; ++j) {
                size == sizeof(double) ?
                (*(CONVERT_DOUBLE_T(SigmaBeginner->ValT) + i + (j + SigmaBeginner->ld[k]) * C)
                         = *(CONVERT_DOUBLE_T(rowBlock[i + k * C]->valBegin) + j)) :
                (*(CONVERT_FLOAT_T(SigmaBeginner->ValT) + i + (j + SigmaBeginner->ld[k]) * C)
                         = *(CONVERT_FLOAT_T(rowBlock[i + k * C]->valBegin) + j));
                SigmaBeginner->ColIndex[i + (j + SigmaBeginner->ld[k]) * C] = rowBlock[i + k * C]->indxBegin[j];
            }
            //*zero += SigmaBeginner->ld[k + 1] - SigmaBeginner->ld[k] - j;
            for (; j < SigmaBeginner->ld[k + 1] - SigmaBeginner->ld[k]; ++j) {
                SigmaBeginner->ColIndex[i + (j + SigmaBeginner->ld[k]) * C] = rowBlock[loc[k] + k * C]->indxBegin[j];
            }
        }
    }
    free(loc);
}

int cmp(const void *A, const void *B) {
    int p = (*((Row_Block_t *) B))->length - (*((Row_Block_t *) A))->length;
    if (p)return -p;
    else {
        return (*((Row_Block_t *) A))->rowNumber - (*((Row_Block_t *) B))->rowNumber;
    }
}


void sell_C_Sigma_get_handle_Selected(spmv_Handle_t handle,
                                      BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE *RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const void *Matrix_Val
) {
    BASIC_INT_TYPE Sigma = C * Times;
    int len;
    if (Sigma == 0) {
        len = 0;
    } else {
        len = m / Sigma;
    }

    int banner = Sigma * len;
    Row_Block_t *rowBlock_ts = NULL;
    Row_Block_t rowBlocks = NULL;
    sigmaEnv_t sigenv = (sigmaEnv_t) malloc(sizeof(Sigma_Block));
    (sigenv)->Sigma = Sigma;
    (sigenv)->C = C;
    (sigenv)->banner = banner;
    BASIC_SIZE_TYPE size = handle->data_size;
    int total = 0;
    int zero = 0;
    if (banner > 0) {
        rowBlock_ts = (Row_Block_t *) malloc(sizeof(Row_Block_t) * banner);
        rowBlocks = (Row_Block_t) malloc(sizeof(struct Row_Block) * banner);
        for (int i = 0; i < banner; ++i) {
            rowBlocks[i].length = RowPtr[i + 1] - RowPtr[i];
            rowBlocks[i].indxBegin = ColIdx + RowPtr[i];
            if (size == sizeof(double)) {
                rowBlocks[i].valBegin =
                        (double *) Matrix_Val + RowPtr[i];
            } else {
                rowBlocks[i].valBegin =
                        (float *) Matrix_Val + RowPtr[i];
            }
            rowBlocks[i].rowNumber = i;
            rowBlock_ts[i] = rowBlocks + i;
        }
/*
        srand(banner);
        for (int i = 0, j; i < banner; ++i) {
            Row_Block_t temp = rowBlock_ts[i];
            j = rand() % banner;
            rowBlock_ts[i] = rowBlock_ts[j];
            rowBlock_ts[j] = temp;
        }

        qsort(rowBlock_ts, banner, sizeof(Row_Block_t), cmp);


        Row_Block_t *Temps = (Row_Block_t *) malloc(sizeof(Row_Block_t) * banner);
        int cnt = 0;
        for (int i = 0; i < Times; ++i) {
            if (i % 2) {
                for (int j = 0; j < len; ++j) {
                    for (int k = 0; k < C; ++k) {
                        Temps[i * C + j * Sigma + k] = rowBlock_ts[cnt++];
                    }
                }
            } else {
                for (int j = len - 1; j >= 0; --j) {
                    for (int k = 0; k < C; ++k) {
                        Temps[i * C + j * Sigma + k] = rowBlock_ts[cnt++];
                    }
                }
            }
        }
        memcpy(rowBlock_ts, Temps, sizeof(Row_Block_t) * banner);
        free(Temps);
        srand(len);

        */
        const int Catch = Sigma;

        (sigenv)->sigmaBlock = (Sigma_Block_t) malloc(sizeof(Sigma_Block) * len);
#pragma omp parallel for
        for (int i = 0; i < len; ++i) {
            spmv_Sigma_Blocks_init((sigenv)->sigmaBlock + i,
                                   C, Sigma, RowPtr,
                                   rowBlock_ts + i * Sigma,
                                   size//, &total, &zero
            );
        }
        /*
        double S = 0;
        double ave = 1.0 * total / len;
        for (int i = 0; i < len; ++i) {
            double cur = handle->sigmaBlock[i].total * C;
            S += (cur - ave) * (cur - ave);
        }
        S = sqrt(S / len) / ave;
        */
        //printf("Sigma,banner,m,C,zero,Average,s,res\n");
        //printf("%d,%d,%d,%d,%f,%f,%f,%f\n", Sigma, banner, m, C, zero * 1.0 / total, ave, S, (m - banner) * 1.0 / m);

        free(rowBlock_ts);
        free(rowBlocks);
    } else {
        (sigenv)->banner = 0;
    }
    handle->extraHandle = sigenv;
}

void spmv_sell_C_Sigma_cpp_d(const spmv_Handle_t handle,
                             BASIC_INT_TYPE m,
                             const BASIC_INT_TYPE *RowPtr,
                             const BASIC_INT_TYPE *ColIdx,
                             const double *Matrix_Val,
                             const double *Vector_Val_X,
                             double *Vector_Val_Y
) {
    if (handle->spmvMethod != Method_SellCSigma) {
        return;
    }
    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY way = handle->vectorizedWay;
    sigmaEnv_t sigenv = (sigmaEnv_t) handle->extraHandle;
    if (sigenv->banner > 0) {
        int C = sigenv->C;
        int Sigma = sigenv->Sigma;
        Sigma_Block_t SigmaBlocks = sigenv->sigmaBlock;
        int length = m / Sigma;
        int C_times = Sigma / C;
        //memset(Vector_Val_Y, 0, size * m);

#pragma omp parallel for
        for (int i = 0; i < length; ++i) {/// sigma
            if (SigmaBlocks[i].ld)
                for (int j = 0; j < C_times; ++j) {

                    basic_d_lineProductGather_avx2(SigmaBlocks[i].ld[j + 1] - SigmaBlocks[i].ld[j],
                                                   C,
                                                   (double *) SigmaBlocks[i].ValT + SigmaBlocks[i].ld[j] * C,
                                                   SigmaBlocks[i].ColIndex + SigmaBlocks[i].ld[j] * C,
                                                   Vector_Val_X,
                                                   SigmaBlocks[i].RowIndex + j * C,
                                                   Vector_Val_Y
                    );
                }

        }
    }

    {
#pragma omp parallel for
        for (int i = sigenv->banner; i < m; ++i) {
            Dot_Product_Avx2_d(RowPtr[i + 1] - RowPtr[i],
                               ColIdx + RowPtr[i],
                               Matrix_Val + RowPtr[i],
                               Vector_Val_X,
                               Vector_Val_Y + i);
        }
    }
}

void spmv_sell_C_Sigma_cpp_s(const spmv_Handle_t handle,
                             BASIC_INT_TYPE m,
                             const BASIC_INT_TYPE *RowPtr,
                             const BASIC_INT_TYPE *ColIdx,
                             const float *Matrix_Val,
                             const float *Vector_Val_X,
                             float *Vector_Val_Y
) {
    if (handle->spmvMethod != Method_SellCSigma) {
        return;
    }
    sigmaEnv_t sigenv = (sigmaEnv_t) handle->extraHandle;
    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY way = handle->vectorizedWay;

    if (sigenv->banner > 0) {
        int C = sigenv->C;
        int Sigma = sigenv->Sigma;
        Sigma_Block_t SigmaBlocks = sigenv->sigmaBlock;
        int length = m / Sigma;
        int C_times = Sigma / C;
        //memset(Vector_Val_Y, 0, size * m);

#pragma omp parallel for
        for (int i = 0; i < length; ++i) {/// sigma
            if (SigmaBlocks[i].ld)
                for (int j = 0; j < C_times; ++j) {

                    basic_s_lineProductGather_avx2(SigmaBlocks[i].ld[j + 1] - SigmaBlocks[i].ld[j],
                                                   C,
                                                   (float *) SigmaBlocks[i].ValT + SigmaBlocks[i].ld[j] * C,
                                                   SigmaBlocks[i].ColIndex + SigmaBlocks[i].ld[j] * C,
                                                   Vector_Val_X,
                                                   SigmaBlocks[i].RowIndex + j * C,
                                                   Vector_Val_Y
                    );
                }

        }
    }

    {
#pragma omp parallel for
        for (int i = sigenv->banner; i < m; ++i) {
            Dot_Product_Avx2_s(RowPtr[i + 1] - RowPtr[i],
                               ColIdx + RowPtr[i],
                               Matrix_Val + RowPtr[i],
                               Vector_Val_X,
                               Vector_Val_Y + i);
        }
    }
}

void spmv_sell_C_Sigma_Selected(const spmv_Handle_t handle,
                                BASIC_INT_TYPE m,
                                const BASIC_INT_TYPE *RowPtr,
                                const BASIC_INT_TYPE *ColIdx,
                                const void *Matrix_Val,
                                const void *Vector_Val_X,
                                void *Vector_Val_Y
) {
    if (handle->data_size == sizeof(double)) {
        spmv_sell_C_Sigma_cpp_d(handle, m, RowPtr, ColIdx, (double *) Matrix_Val, (double *) Vector_Val_X,
                                (double *) Vector_Val_Y);
    } else {
        spmv_sell_C_Sigma_cpp_s(handle, m, RowPtr, ColIdx, (float *) Matrix_Val, (float *) Vector_Val_X,
                                (float *) Vector_Val_Y);
    }
}