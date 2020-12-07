//
// Created by kouushou on 2020/12/4.
//
#include <gemv.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
typedef struct Row_Block {
    const GEMV_INT_TYPE   *indxBegin;
    const GEMV_VAL_TYPE   *valBegin;
    GEMV_INT_TYPE   length;
    GEMV_INT_TYPE   rowNumber;
}Row_Block,*Row_Block_t;

int cmp(const void* A,const void* B){
    int p =(*((Row_Block_t*)B))->length-(*((Row_Block_t*)A))->length;
    if(p)return p;
    else{
        return (*((Row_Block_t*)A))->rowNumber-(*((Row_Block_t*)B))->rowNumber;
    }
}

void sell_C_Sigma_get_handle(gemv_Handle_t handle,
                             GEMV_INT_TYPE Times,GEMV_INT_TYPE C,
                             GEMV_INT_TYPE m,
                             const GEMV_INT_TYPE*RowPtr,
                             const GEMV_INT_TYPE*ColIdx,
                             const GEMV_VAL_TYPE*Matrix_Val,
                             GEMV_INT_TYPE nnzR,
                             GEMV_INT_TYPE nthreads) {
    GEMV_INT_TYPE Sigma = C * Times;
    int len = m / Sigma;
    int total_len = Sigma*len;
    Row_Block_t *rowBlock_ts = NULL;
    Row_Block_t rowBlocks = NULL;
    if (total_len > 0) {
        rowBlock_ts = (Row_Block_t *) malloc(sizeof(Row_Block_t) * total_len);
        rowBlocks = (Row_Block_t)malloc(sizeof(struct Row_Block)*total_len);
        for(int i = 0 ; i < total_len ; ++i){
            rowBlocks[i].length = RowPtr[i+1]-RowPtr[i];
            rowBlocks[i].indxBegin = ColIdx+RowPtr[i];
            rowBlocks[i].valBegin = Matrix_Val+RowPtr[i];
            rowBlocks[i].rowNumber = i;
            rowBlock_ts[i] = rowBlocks+i;
        }
        //qsort(rowBlock_ts, m, sizeof(Row_Block_t), cmp);
        for(int i = 0 ,I_of_Sigma = 0; i < len ; ++i,I_of_Sigma+=Sigma){
            qsort(rowBlock_ts+I_of_Sigma, Sigma, sizeof(Row_Block_t), cmp);
        }



    }
}
void sell_C_Sigma_gemv(const gemv_Handle_t handle,
                       GEMV_INT_TYPE m,
                       const GEMV_INT_TYPE* RowPtr,
                       const GEMV_INT_TYPE* ColIdx,
                       const GEMV_VAL_TYPE* Matrix_Val,
                       const GEMV_VAL_TYPE* Vector_Val_X,
                       GEMV_VAL_TYPE*       Vector_Val_Y){

}