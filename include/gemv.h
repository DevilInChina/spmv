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
void gemv_destory_handle(gemv_Handle_t this_handle);


/**
 * @brief clear handle for the next usage, designed for developers of lib
 * @param this_handle
 */
void gemv_clear_handle(gemv_Handle_t this_handle);



void spmv_create_handle_all_in_one(gemv_Handle_t *Handle,
                                   BASIC_INT_TYPE m,
                                   const BASIC_INT_TYPE*RowPtr,
                                   const BASIC_INT_TYPE *ColIdx,
                                   const void *Matrix_Val,
                                   BASIC_SIZE_TYPE nthreads,
                                   STATUS_GEMV_HANDLE Function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay
                        );


void spmv(const gemv_Handle_t handle,
          BASIC_INT_TYPE m,
          const BASIC_INT_TYPE* RowPtr,
          const BASIC_INT_TYPE* ColIdx,
          const void* Matrix_Val,
          const void* Vector_Val_X,
          void*       Vector_Val_Y);

#endif