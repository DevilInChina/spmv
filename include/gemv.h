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
 * @brief create a empty handle with initialize
 * @return
 */
gemv_Handle_t gemv_create_handle();

/**
 * @brief clear handle for the next usage, designed for developers of lib
 * @param this_handle
 */
void gemv_clear_handle(gemv_Handle_t this_handle);

/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param nnzR
 * @param nthreads
 */
void parallel_balanced_get_handle(
        gemv_Handle_t* handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        BASIC_INT_TYPE nnzR,
        BASIC_INT_TYPE nthreads);

/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param nnzR
 * @param nthreads
 */
void parallel_balanced2_get_handle(
        gemv_Handle_t* handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        BASIC_INT_TYPE nnzR,
        BASIC_INT_TYPE nthreads);

void sell_C_Sigma_get_handle_Selected(gemv_Handle_t* handle,
                                      BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE*RowPtr,
                                      const BASIC_INT_TYPE*ColIdx,
                                      const void*Matrix_Val,
                                      BASIC_INT_TYPE nthreads,
                                      BASIC_SIZE_TYPE size
) ;

void sell_C_Sigma_get_handle_s(gemv_Handle_t* handle,
                               BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                               BASIC_INT_TYPE m,
                               const BASIC_INT_TYPE*RowPtr,
                               const BASIC_INT_TYPE*ColIdx,
                               const float *Matrix_Val,
                               BASIC_INT_TYPE nthreads
);


void sell_C_Sigma_get_handle_d(gemv_Handle_t* handle,
                               BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                               BASIC_INT_TYPE m,
                               const BASIC_INT_TYPE*RowPtr,
                               const BASIC_INT_TYPE*ColIdx,
                               const double *Matrix_Val,
                               BASIC_INT_TYPE nthreads
);


void spmv_serial_Selected(BASIC_INT_TYPE m,
                          const BASIC_INT_TYPE*RowPtr,
                          const BASIC_INT_TYPE *ColIdx,
                          const void *Matrix_Val,
                          const void*Vector_Val_X,
                          void *Vector_Val_Y,
                          BASIC_SIZE_TYPE size,
                          VECTORIZED_WAY vectorizedWay);

void spmv_parallel_Selected(BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE*RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const void *Matrix_Val,
                            const void*Vector_Val_X,
                            void *Vector_Val_Y,
                            BASIC_SIZE_TYPE size,
                            VECTORIZED_WAY vectorizedWay
);


void spmv_parallel_balanced_Selected(
        const gemv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y,
        BASIC_SIZE_TYPE size,
        VECTORIZED_WAY way
        );

void spmv_parallel_balanced2_Selected(
        const gemv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y,
        BASIC_SIZE_TYPE size,
        VECTORIZED_WAY way
);


void spmv_sell_C_Sigma_Selected(const gemv_Handle_t handle,
                                BASIC_INT_TYPE m,
                                const BASIC_INT_TYPE* RowPtr,
                                const BASIC_INT_TYPE* ColIdx,
                                const void* Matrix_Val,
                                const void* Vector_Val_X,
                                void*       Vector_Val_Y,
                                BASIC_SIZE_TYPE size,
                                VECTORIZED_WAY way
);

typedef void(*spmv_no_handle_s_function)(BASIC_INT_TYPE m,
                                         const BASIC_INT_TYPE*RowPtr,
                                         const BASIC_INT_TYPE *ColIdx,
                                         const float *Matrix_Val,
                                         const float *Vector_Val_X,
                                         float *Vector_Val_Y);

typedef void(*spmv_no_handle_d_function)(BASIC_INT_TYPE m,
                                         const BASIC_INT_TYPE*RowPtr,
                                         const BASIC_INT_TYPE *ColIdx,
                                         const double *Matrix_Val,
                                         const double *Vector_Val_X,
                                         double *Vector_Val_Y);


typedef void(*spmv_handle_s_function)(const gemv_Handle_t handle,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE*RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const float *Matrix_Val,
                                      const float *Vector_Val_X,
                                      float *Vector_Val_Y);

typedef void(*spmv_handle_d_function)(const gemv_Handle_t handle,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE*RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const double *Matrix_Val,
                                      const double *Vector_Val_X,
                                      double *Vector_Val_Y);


typedef void(*spmv_handle_function)(const gemv_Handle_t handle,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE*RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const void *Matrix_Val,
                                      const void *Vector_Val_X,
                                      void *Vector_Val_Y,
                                      BASIC_SIZE_TYPE size,
                                      VECTORIZED_WAY vectorizedWay
                                      );
extern const spmv_handle_function spmvs[];

FUNC_DECLARES(FUNC_NO_HANDLE_DECLARES,serial);

FUNC_DECLARES(FUNC_NO_HANDLE_DECLARES,parallel);

FUNC_DECLARES(FUNC_HANDLE_DECLARES,sell_C_Sigma);

FUNC_DECLARES(FUNC_HANDLE_DECLARES,parallel_balanced);

FUNC_DECLARES(FUNC_HANDLE_DECLARES,parallel_balanced2);



#endif