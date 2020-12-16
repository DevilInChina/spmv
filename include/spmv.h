#ifndef GEMV_GEMV_H_
#define GEMV_GEMV_H_
#include <omp.h>
#include "lineProduct.h"
#include "dotProduct.h"
#include <immintrin.h>

#define ALIGENED_SIZE 32

/**
 * @brief destroy handle, null with doing nothing
 * @param this_handle
 * @return
 */
void spmv_destory_handle(spmv_Handle_t this_handle);


/**
 * @brief clear handle for the next usage, designed for developers of lib
 * @param this_handle
 */
void spmv_clear_handle(spmv_Handle_t this_handle);


/**
 * @brief initialize a handle according to parameters sent in ,
 * @param Handle
 * @param m                 rows of the csr-storage matrix
 * @param RowPtr            length is (m+1) , RowPtr[i]-RowPtr[i-1] means the number of non-zero element at line i
 * @param ColIdx            length is (RowPtr[m]-RowPtr[0]) , means Collum index of each non-zero element
 * @param Matrix_Val        (void*) length is (RowPtr[m]-RowPtr[0]) , means Value of each non-zero element
 * @param nthreads          max number of threads can be use
 * @param Function          calculate way (serial,parallel,parallel_balanced,parallel_balanced2,sell_C_Sigma)
 * @param size              sizeof(double)/sizeof(float) refer to call float or double version
 * @param vectorizedWay     using (not use,avx2,avx512)
 */
void spmv_create_handle_all_in_one(spmv_Handle_t *Handle,
                                   BASIC_INT_TYPE m,
                                   const BASIC_INT_TYPE *RowPtr,
                                   const BASIC_INT_TYPE *ColIdx,
                                   const void *Matrix_Val,
                                   BASIC_SIZE_TYPE nthreads,
                                   SPMV_METHODS Function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay
                        );


/**
 * @brief calculate according to handle
 * @param handle            must call function spmv_create_handle_all_in_one before
 * @param m                 rows of the csr-storage matrix
 * @param RowPtr            length is (m+1) , RowPtr[i]-RowPtr[i-1] means the number of non-zero element at line i
 * @param ColIdx            length is (RowPtr[m]-RowPtr[0]) , means Collum index of each non-zero element
 * @param Matrix_Val        (void*) length is (RowPtr[m]-RowPtr[0]) , means Value of each non-zero element
 * @param Vector_Val_X      length is n or max(ColIdx) (max element in ColIdx)
 * @param Vector_Val_Y      length is m
 */
void spmv(const spmv_Handle_t handle,
          BASIC_INT_TYPE m,
          const BASIC_INT_TYPE *RowPtr,
          const BASIC_INT_TYPE *ColIdx,
          const void* Matrix_Val,
          const void* Vector_Val_X,
          void*       Vector_Val_Y);

#endif