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
 * @param m
 * @param RowPtr
 * @param nnzR
 * @param nthreads
 * @param csrSplitter
 */
void parallel_balanced_get_csrSplitter(
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE*RowPtr,
        GEMV_INT_TYPE nnzR,
        GEMV_INT_TYPE nthreads,
        GEMV_INT_TYPE**csrSplitter);

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
                            GEMV_INT_TYPE        nthreads,
                            const GEMV_INT_TYPE* csrSplitter,
                            GEMV_INT_TYPE m,
                            const GEMV_INT_TYPE* RowPtr,
                            const GEMV_INT_TYPE* ColIdx,
                            const GEMV_VAL_TYPE* Matrix_Val,
                            const GEMV_VAL_TYPE* Vector_Val_X,
                            GEMV_VAL_TYPE*Vector_Val_Y);


/**
 *
 * @param nthreads
 * @param Yid
 * @param Apinter
 * @param Start1
 * @param End1
 * @param Start2
 * @param End2
 * @param csrSplitter
 * @param m
 * @param RowPtr
 * @param ColIdx
 * @param Matrix_Val
 * @param Vector_Val_X
 * @param Vector_Val_Y
 */
void parallel_balanced2_gemv(
        GEMV_INT_TYPE        nthreads,
        const GEMV_INT_TYPE* Yid,
        const GEMV_INT_TYPE* Apinter,
        const GEMV_INT_TYPE* Start1,
        const GEMV_INT_TYPE* End1,
        const GEMV_INT_TYPE* Start2,
        const GEMV_INT_TYPE* End2,
        const GEMV_INT_TYPE* csrSplitter,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y);
#endif