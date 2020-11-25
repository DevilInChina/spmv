#ifndef __GEMV_H_
#define __GEMV_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>

#ifndef GEMV_INT_TYPE
#define GEMV_INT_TYPE int
#endif

#ifndef GEMV_VAL_TYPE
#define GEMV_VAL_TYPE float
#endif
typedef enum STATUS_GEMV_HANDLE{
    NONE,
    BALANCED,
    BALANCED2
}STATUS_GEMV_HANDLE;
typedef struct gemv_Handle gemv_Handle;
typedef gemv_Handle*  gemv_Handle_t;
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

/**
 *
 * @param nthreads
 * @param csrSplitter
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
void parallel_balanced_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);
#endif