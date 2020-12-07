#ifndef GEMV_GEMV_H_
#define GEMV_GEMV_H_
#include <omp.h>
#include "lineProduct.h"
#include <immintrin.h>


float gemv_s_dotProduct(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X);

float gemv_s_dotProduct_avx2(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X);

float gemv_s_dotProduct_avx512(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X);


double gemv_d_dotProduct(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const double *Val,const double *X) ;

double gemv_d_dotProduct_avx2(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const double *Val,const double *X) ;

double gemv_d_dotProduct_avx512(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const double *Val,const double *X);

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
 * @introduce Simple calculate matrix storage by csr product vector X, result store in Y
 * @author
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void serial_gemv(GEMV_INT_TYPE m,
                 const GEMV_INT_TYPE*RowPtr,
                 const GEMV_INT_TYPE *ColIdx,
                 const GEMV_VAL_TYPE*Matrix_Val,
                 const GEMV_VAL_TYPE*Vector_Val_X,
                 GEMV_VAL_TYPE*Vector_Val_Y);

/**
 *
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_gemv(GEMV_INT_TYPE m,
                 const GEMV_INT_TYPE *RowPtr,
                 const GEMV_INT_TYPE *ColIdx,
                 const GEMV_VAL_TYPE *Matrix_Val,
                 const GEMV_VAL_TYPE *Vector_Val_X,
                 GEMV_VAL_TYPE       *Vector_Val_Y);

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
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE*RowPtr,
        GEMV_INT_TYPE nnzR,
        GEMV_INT_TYPE nthreads);

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
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE*RowPtr,
        GEMV_INT_TYPE nnzR,
        GEMV_INT_TYPE nthreads);


void parallel_balanced_gemv_Selected(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y,
        DOT_PRODUCT_WAY way
);

void parallel_balanced2_gemv_Selected(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y,
        DOT_PRODUCT_WAY way
);


/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced_gemv(
                            gemv_Handle_t handle,
                            GEMV_INT_TYPE m,
                            const GEMV_INT_TYPE* RowPtr,
                            const GEMV_INT_TYPE* ColIdx,
                            const GEMV_VAL_TYPE* Matrix_Val,
                            const GEMV_VAL_TYPE* Vector_Val_X,
                            GEMV_VAL_TYPE*Vector_Val_Y);

/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);


/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced_gemv_avx512(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);


/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced2_gemv(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);

/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced2_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);

/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced2_gemv_avx512(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);


GEMV_VAL_TYPE (*inner__gemv_GetDotProduct(size_t types,DOT_PRODUCT_WAY way))
        (GEMV_INT_TYPE len, const GEMV_INT_TYPE *indx,
         const GEMV_VAL_TYPE *Val, const GEMV_VAL_TYPE *X);


void sell_C_Sigma_get_handle(gemv_Handle_t* handle,
                             GEMV_INT_TYPE Times,GEMV_INT_TYPE C,
                             GEMV_INT_TYPE m,
                             const GEMV_INT_TYPE*RowPtr,
                             const GEMV_INT_TYPE*ColIdx,
                             const GEMV_VAL_TYPE*Matrix_Val,
                             GEMV_INT_TYPE nnzR,
                             GEMV_INT_TYPE nthreads);
#endif