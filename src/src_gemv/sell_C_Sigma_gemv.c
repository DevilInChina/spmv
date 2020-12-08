//
// Created by kouushou on 2020/12/4.
//
#include <gemv.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))

void C_Block_init(C_Block_t this_block,int C,const GEMV_INT_TYPE *RowPtr,Row_Block_t *rowBlock) {
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

    this_block->ValT = aligned_alloc(ALIGENED_SIZE, sizeof(GEMV_VAL_TYPE) * C * maxs);
    this_block->ColIndex = aligned_alloc(ALIGENED_SIZE, sizeof(GEMV_INT_TYPE) * C * maxs);
    this_block->RowIndex = aligned_alloc(ALIGENED_SIZE, sizeof(GEMV_INT_TYPE) * C);
    this_block->Y = aligned_alloc(ALIGENED_SIZE,sizeof(GEMV_VAL_TYPE)*C);

    for (int i = 0; i < C; ++i) {
        int j = 0;
        for (; j < RowPtr[rowBlock[i]->rowNumber + 1] - RowPtr[rowBlock[i]->rowNumber]; ++j) {
            this_block->ValT[i + j * C] = rowBlock[i]->valBegin[j];
            this_block->ColIndex[i + j * C] = rowBlock[i]->indxBegin[j];
        }
        for (; j < maxs; ++j) {
            this_block->ValT[i + j * C] = 0;
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



void sell_C_Sigma_get_handle(gemv_Handle_t* handle,
                             GEMV_INT_TYPE Times,GEMV_INT_TYPE C,
                             GEMV_INT_TYPE m,
                             const GEMV_INT_TYPE*RowPtr,
                             const GEMV_INT_TYPE*ColIdx,
                             const GEMV_VAL_TYPE*Matrix_Val,
                             GEMV_INT_TYPE nthreads) {
    GEMV_INT_TYPE Sigma = C * Times;
    int len = m / Sigma;
    int banner = Sigma * len;
    Row_Block_t *rowBlock_ts = NULL;
    Row_Block_t rowBlocks = NULL;
    *handle = gemv_create_handle();

    (*handle)->nthreads = nthreads;
    (*handle)->status = STATUS_BALANCED;
    (*handle)->Sigma = Sigma;
    (*handle)->C = C;
    (*handle)->banner = banner;

    if (banner > 0) {
        rowBlock_ts = (Row_Block_t *) malloc(sizeof(Row_Block_t) * banner);
        rowBlocks = (Row_Block_t) malloc(sizeof(struct Row_Block) * banner);
        for (int i = 0; i < banner; ++i) {
            rowBlocks[i].length = RowPtr[i + 1] - RowPtr[i];
            rowBlocks[i].indxBegin = ColIdx + RowPtr[i];
            rowBlocks[i].valBegin = Matrix_Val + RowPtr[i];
            rowBlocks[i].rowNumber = i;
            rowBlock_ts[i] = rowBlocks + i;
        }
        //qsort(rowBlock_ts, m, sizeof(Row_Block_t), cmp);
        for (int i = 0, I_of_Sigma = 0; i < len; ++i, I_of_Sigma += Sigma) {
            qsort(rowBlock_ts + I_of_Sigma, Sigma, sizeof(Row_Block_t), cmp);
        }

        int siz = (*handle)->banner / C;
        (*handle)->C_Blocks = (C_Block_t) malloc(sizeof(C_Block) * siz);

        for (int CBlock = 0; CBlock < (*handle)->banner; CBlock += C) {
            C_Block_init((*handle)->C_Blocks + CBlock / C, C, RowPtr,
                         rowBlock_ts + CBlock);
        }
        free(rowBlock_ts);
        free(rowBlocks);
    }else{
        (*handle)->banner = 0;
    }

}



void sell_C_Sigma_gemv(const gemv_Handle_t handle,
                                GEMV_INT_TYPE m,
                                const GEMV_INT_TYPE* RowPtr,
                                const GEMV_INT_TYPE* ColIdx,
                                const GEMV_VAL_TYPE* Matrix_Val,
                                const GEMV_VAL_TYPE* Vector_Val_X,
                                GEMV_VAL_TYPE*       Vector_Val_Y
){
    sell_C_Sigma_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_NONE);
}

void sell_C_Sigma_gemv_avx2(const gemv_Handle_t handle,
                       GEMV_INT_TYPE m,
                       const GEMV_INT_TYPE* RowPtr,
                       const GEMV_INT_TYPE* ColIdx,
                       const GEMV_VAL_TYPE* Matrix_Val,
                       const GEMV_VAL_TYPE* Vector_Val_X,
                       GEMV_VAL_TYPE*       Vector_Val_Y
){
    sell_C_Sigma_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_AVX2);
}

void sell_C_Sigma_gemv_avx512(const gemv_Handle_t handle,
                       GEMV_INT_TYPE m,
                       const GEMV_INT_TYPE* RowPtr,
                       const GEMV_INT_TYPE* ColIdx,
                       const GEMV_VAL_TYPE* Matrix_Val,
                       const GEMV_VAL_TYPE* Vector_Val_X,
                       GEMV_VAL_TYPE*       Vector_Val_Y
){
    sell_C_Sigma_gemv_Selected(handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y,DOT_AVX512);
}