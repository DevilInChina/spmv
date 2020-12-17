//
// Created by kouushou on 2020/12/4.
//
#include "inner_spmv.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX(a,b) ((a)>(b)?(a):(b))

void gemv_C_Block_init(C_Block_t this_block, int C,
                       const BASIC_INT_TYPE *RowPtr, Row_Block_t *rowBlock,
                       BASIC_SIZE_TYPE size,int *total,int *zero
                       ) {
    int maxs = 0;
    for (int i = 0; i < C; ++i) {
        maxs = MAX(maxs, RowPtr[rowBlock[i]->rowNumber + 1] - RowPtr[rowBlock[i]->rowNumber]);

    }

    this_block->C = C;
    this_block->ld = maxs;
    if (maxs == 0) {
        this_block->ValT = NULL;
        this_block->RowIndex = NULL;
        this_block->ColIndex = NULL;
        this_block->Y = NULL;
        return;
    }

    this_block->ValT = aligned_alloc(ALIGENED_SIZE, size * C * maxs);
    this_block->ColIndex = aligned_alloc(ALIGENED_SIZE, sizeof(BASIC_INT_TYPE) * C * maxs);
    this_block->RowIndex = aligned_alloc(ALIGENED_SIZE, sizeof(BASIC_INT_TYPE) * C);
    this_block->Y = aligned_alloc(ALIGENED_SIZE,size*C);
    *total+=C*maxs;
    for (int i = 0; i < C; ++i) {
        int j = 0;
        for (; j < RowPtr[rowBlock[i]->rowNumber + 1] - RowPtr[rowBlock[i]->rowNumber]; ++j) {
            size==sizeof(double )?
            (*(CONVERT_DOUBLE_T(this_block->ValT)+i + j * C)
                            = *(CONVERT_DOUBLE_T(rowBlock[i]->valBegin)+j)):
            (*(CONVERT_FLOAT_T(this_block->ValT)+i + j * C)
                            = *(CONVERT_FLOAT_T(rowBlock[i]->valBegin)+j));

            this_block->ColIndex[i + j * C] = rowBlock[i]->indxBegin[j];
        }
        for (; j < maxs; ++j) {

            size==sizeof(double )?
            (*(CONVERT_DOUBLE_T(this_block->ValT)+i + j * C)
                     = 0):
            (*(CONVERT_FLOAT_T(this_block->ValT)+i + j * C)
                     = 0);
            ++(*zero);
            this_block->ColIndex[i + j * C] = 0;
        }
        this_block->RowIndex[i] = rowBlock[i]->rowNumber;
    }
}

int cmp(const void* A,const void* B){
    int p =(*((Row_Block_t*)B))->length-(*((Row_Block_t*)A))->length;
    if(p)return p;
    else{
        return (*((Row_Block_t*)A))->rowNumber-(*((Row_Block_t*)B))->rowNumber;
    }
}



void sell_C_Sigma_get_handle_Selected(spmv_Handle_t handle,
                                      BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE*RowPtr,
                                      const BASIC_INT_TYPE*ColIdx,
                                      const void*Matrix_Val
                             ) {
    BASIC_INT_TYPE Sigma = C * Times;
    int len = m / Sigma;
    int banner = Sigma * len;
    Row_Block_t *rowBlock_ts = NULL;
    Row_Block_t rowBlocks = NULL;

    (handle)->Sigma = Sigma;
    (handle)->C = C;
    (handle)->banner = banner;
    BASIC_SIZE_TYPE size = handle->data_size;
    int total = 0;
    int zero = 0;
    if (banner > 0) {
        rowBlock_ts = (Row_Block_t *) malloc(sizeof(Row_Block_t) * banner);
        rowBlocks = (Row_Block_t) malloc(sizeof(struct Row_Block) * banner);
        for (int i = 0; i < banner; ++i) {
            rowBlocks[i].length = RowPtr[i + 1] - RowPtr[i];
            rowBlocks[i].indxBegin = ColIdx + RowPtr[i];
            rowBlocks[i].valBegin = Matrix_Val + RowPtr[i]*size;
            rowBlocks[i].rowNumber = i;
            rowBlock_ts[i] = rowBlocks + i;
        }
        //qsort(rowBlock_ts, m, sizeof(Row_Block_t), cmp);
        for (int i = 0, I_of_Sigma = 0; i < len; ++i, I_of_Sigma += Sigma) {
            qsort(rowBlock_ts + I_of_Sigma, Sigma, sizeof(Row_Block_t), cmp);
        }

        int siz = (handle)->banner / C;
        (handle)->C_Blocks = (C_Block_t) malloc(sizeof(C_Block) * siz);

        for (int CBlock = 0; CBlock < (handle)->banner; CBlock += C) {
            gemv_C_Block_init((handle)->C_Blocks + CBlock / C, C, RowPtr,
                              rowBlock_ts + CBlock,size,&total,&zero);
        }
        printf("%d %.5f%%\n",C,zero*100.0/total);
        free(rowBlock_ts);
        free(rowBlocks);
    }else{
        (handle)->banner = 0;
    }
}

void spmv_sell_C_Sigma_Selected(const spmv_Handle_t handle,
                                BASIC_INT_TYPE m,
                                const BASIC_INT_TYPE* RowPtr,
                                const BASIC_INT_TYPE* ColIdx,
                                const void* Matrix_Val,
                                const void* Vector_Val_X,
                                void*       Vector_Val_Y
){
    if(handle->spmvMethod != Method_SellCSigma){
        return;
    }
    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY way = handle->vectorizedWay;

    dot_product_function
    dot_product = inner_basic_GetDotProduct(size);

    packLine_product_function
    packLine_product = inner_basic_GetPackLineProduct(size);

    gather_function
    gather = inner_basic_GetGather(size);

    if(handle->banner>0) {
        int C = handle->C;
        int Sigma = handle->Sigma;
        C_Block_t cBlocks = handle->C_Blocks;
        int length = m / Sigma;
        int C_times = Sigma / C;
        memset(Vector_Val_Y, 0, size * m);

#pragma omp parallel for
        for (int i = 0; i < length; ++i) {/// sigma
            int SigmaBlock = i * C_times;

            for (int j = 0; j < C_times; ++j) {
                memset(cBlocks[j + SigmaBlock].Y, 0, size * cBlocks[j + SigmaBlock].C);
                packLine_product(cBlocks[j + SigmaBlock].ld, C, cBlocks[j + SigmaBlock].ValT,
                                 cBlocks[j + SigmaBlock].ColIndex,
                                 Vector_Val_X, cBlocks[j + SigmaBlock].Y, way
                );
                gather(cBlocks[j + SigmaBlock].C,
                       cBlocks[j + SigmaBlock].Y,
                       cBlocks[j + SigmaBlock].RowIndex, Vector_Val_Y, way);
            }
        }
    }

    {
#pragma omp parallel for
        for (int i = handle->banner; i < m; ++i) {
            dot_product(RowPtr[i+1]-RowPtr[i],
                        ColIdx+RowPtr[i],Matrix_Val+RowPtr[i]*size,Vector_Val_X,Vector_Val_Y+i*size,way);
        }
    }
}



