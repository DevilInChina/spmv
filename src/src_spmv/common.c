//
// Created by kouushou on 2020/11/25.
//

#include "inner_spmv.h"
#include <math.h>
#include <string.h>
/**
 * @brief init parameters used in balanced and balanced2
 * @param this_handle
 */


void init_sell_C_Sigma(spmv_Handle_t this_handle){
    this_handle->Sigma = 0;
    this_handle->C = 0;
    this_handle->banner = 0;

    this_handle->sigmaBlock = NULL;
}

/**
 * @brief free parameters used in balanced and balanced2
 * @param this_handle
 */


void C_Block_destory(Sigma_Block_t this_block){
    free(this_block->RowIndex);
    free(this_block->ColIndex);
    free(this_block->ValT);
    free(this_block->Y);
    free(this_block->ld);
}

void clear_Sell_C_Sigma(spmv_Handle_t this_handle) {
    if(this_handle && this_handle->spmvMethod==Method_SellCSigma) {
        if (this_handle->Sigma) {
            int siz = this_handle->banner / this_handle->Sigma;
            for (int i = 0; i < siz; ++i) {
                C_Block_destory(this_handle->sigmaBlock + i);
            }
        }
        free(this_handle->sigmaBlock);
    }
}

void gemv_Handle_init(spmv_Handle_t this_handle){
    this_handle->spmvMethod = Method_Serial;
    this_handle->nthreads = 0;
    this_handle->extraHandle = NULL;

    init_sell_C_Sigma(this_handle);

}

void gemv_Handle_clear(spmv_Handle_t this_handle) {

    balancedHandleDestroy(this_handle);

    balanced2HandleDestroy(this_handle);

    clear_Sell_C_Sigma(this_handle);

    csr5HandleDestory(this_handle);

    numaHandleDestory(this_handle);

    gemv_Handle_init(this_handle);
}

void spmv_destory_handle(spmv_Handle_t this_handle){
    gemv_Handle_clear(this_handle);
    free(this_handle);
}

spmv_Handle_t gemv_create_handle(){
    spmv_Handle_t ret = malloc(sizeof(spmv_Handle));
    gemv_Handle_init(ret);
    return ret;
}

void spmv_clear_handle(spmv_Handle_t this_handle){
    gemv_Handle_clear(this_handle);
}






void handle_init_common_parameters(spmv_Handle_t this_handle,
                                   BASIC_SIZE_TYPE nthreads,
                                   SPMV_METHODS function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay){
    this_handle->nthreads = nthreads;
    this_handle->vectorizedWay = vectorizedWay;
    this_handle->data_size = size;
    this_handle->spmvMethod = function;
}

const spmv_function spmv_functions[] = {
        spmv_serial_Selected,
        spmv_parallel_Selected,
        spmv_parallel_balanced_Selected,
        spmv_parallel_balanced2_Selected,
        spmv_sell_C_Sigma_Selected,
        spmv_csr5Spmv_Selected,
        spmv_numa_Selected
};

void spmv_create_handle_all_in_one(spmv_Handle_t *Handle,
                                   BASIC_INT_TYPE m,
                                   BASIC_INT_TYPE n,
                                   const BASIC_INT_TYPE*RowPtr,
                                   const BASIC_INT_TYPE *ColIdx,
                                   const void *Matrix_Val,
                                   BASIC_SIZE_TYPE nthreads,
                                   SPMV_METHODS Function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay
){
    *Handle = gemv_create_handle();
    if(Function < Method_Serial || Function >= Method_Total_Size)Function = Method_Serial;

    handle_init_common_parameters(*Handle,nthreads,Function,size,vectorizedWay);
    const int Sigma = 7744;
    int C = (sizeof(double )/size)<<(vectorizedWay+1);
    const int Turn = 8;
    const int Times = m/nthreads/C;
    switch (Function) {
        case Method_Balanced:{
            parallel_balanced_get_handle(*Handle,m,RowPtr,RowPtr[m]-RowPtr[0]);
        }break;
        case Method_Balanced2:{
            parallel_balanced2_get_handle(*Handle,m,RowPtr,RowPtr[m]-RowPtr[0]);
        }break;
        case Method_SellCSigma:{
            sell_C_Sigma_get_handle_Selected(*Handle,Times,C,m,RowPtr,ColIdx,Matrix_Val);
        }break;
        case Method_CSR5SPMV: {
            if (size == sizeof(double)) {
                csr5Spmv_get_handle_Selected(*Handle, m, n, (int *) RowPtr, (int *) ColIdx, Matrix_Val);
            }else{
                (*Handle)->spmvMethod = Method_SellCSigma;
                sell_C_Sigma_get_handle_Selected(*Handle,Times
                                                 ,C,m,RowPtr,ColIdx,Matrix_Val);
            }
        }break;
        case Method_Numa:{
            int k = numa_spmv_get_handle_Selected(*Handle,m,n,(int *) RowPtr, (int *) ColIdx, Matrix_Val);
            if(k==0){
                (*Handle)->spmvMethod = Method_Balanced2;
                parallel_balanced2_get_handle(*Handle,m,RowPtr,RowPtr[m]-RowPtr[0]);
            }
        }
        default:{

            return;
        }
    }

}

void inner_exclusive_scan(BASIC_INT_TYPE *input, int length) {
    if (length == 0 || length == 1)
        return;

    BASIC_INT_TYPE old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++) {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

void inner_matrix_transposition_d(const int           m,
                          const int           n,
                          const BASIC_INT_TYPE     nnz,
                          const BASIC_INT_TYPE    *csrRowPtr,
                          const int          *csrColIdx,
                          const double *csrVal,
                          int          *cscRowIdx,
                                BASIC_INT_TYPE    *cscColPtr,
                          double *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(BASIC_INT_TYPE) * (n+1));
    for (BASIC_INT_TYPE i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    inner_exclusive_scan(cscColPtr, n + 1);

    BASIC_INT_TYPE *cscColIncr = (BASIC_INT_TYPE *)malloc(sizeof(BASIC_INT_TYPE) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(BASIC_INT_TYPE) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (BASIC_INT_TYPE j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}


void inner_matrix_transposition_s(const int           m,
                                  const int           n,
                                  const BASIC_INT_TYPE     nnz,
                                  const BASIC_INT_TYPE    *csrRowPtr,
                                  const int          *csrColIdx,
                                  const float *csrVal,
                                  int          *cscRowIdx,
                                  BASIC_INT_TYPE    *cscColPtr,
                                  float *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(BASIC_INT_TYPE) * (n+1));
    for (BASIC_INT_TYPE i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    inner_exclusive_scan(cscColPtr, n + 1);

    BASIC_INT_TYPE *cscColIncr = (BASIC_INT_TYPE *)malloc(sizeof(BASIC_INT_TYPE) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(BASIC_INT_TYPE) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (BASIC_INT_TYPE j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}

void spmv(const spmv_Handle_t handle,
          BASIC_INT_TYPE m,
          const BASIC_INT_TYPE* RowPtr,
          const BASIC_INT_TYPE* ColIdx,
          const void* Matrix_Val,
          const void* Vector_Val_X,
          void*       Vector_Val_Y){
    if(handle==NULL)return;
    spmv_functions[handle->spmvMethod](handle, m, RowPtr, ColIdx, Matrix_Val, Vector_Val_X, Vector_Val_Y);
}
#define SINGLE(arg) #arg
#define STR(args1,args2) #args1 #args2
#define VEC_STRING(NAME)\
STR(NAME,_VECTOR_NONE),\
STR(NAME,_VECTOR_AVX2),\
STR(NAME,_VECTOR_AVX512)

#define ALL_FUNC_SRTING \
VEC_STRING(Method_Serial),\
VEC_STRING(Method_Parallel),\
VEC_STRING(Method_Balanced),\
VEC_STRING(Method_Balanced2),\
VEC_STRING(Method_SellCSigma),\
VEC_STRING(Method_Csr5Spmv)//,\
//VEC_STRING(Method_NumaSpmv)

const char * funcNames[]= {
    ALL_FUNC_SRTING
};
const char*Methods_names[]={
        SINGLE(Method_Serial),
        SINGLE(Method_Parallel),
        SINGLE(Method_Balanced),
        SINGLE(Method_Balanced2),
        SINGLE(Method_SellCSigma),
        SINGLE(Method_Csr5Spmv)
        //,SINGLE(Method_NumaSpmv)
};
const char*Vectorized_names[]={
        SINGLE(VECTOR_NONE),
        SINGLE(VECTOR_AVX2),
        SINGLE(VECTOR_AVX512),
};