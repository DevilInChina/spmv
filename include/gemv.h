#ifndef __GEMV_H_
#define __GEMV_H_
#include <omp.h>
#include <immintrin.h>

#ifndef GEMV_INT_TYPE
#define GEMV_INT_TYPE int
#endif

#ifndef GEMV_VAL_TYPE
#define GEMV_VAL_TYPE float
#endif


typedef enum DOT_PRODUCT_WAY{
    DOT_NONE,
    DOT_AVX2,
    DOT_AVX512
}DOT_PRODUCT_WAY;


typedef enum STATUS_GEMV_HANDLE{
    NONE,
    BALANCED,
    BALANCED2
}STATUS_GEMV_HANDLE;
typedef struct gemv_Handle {
    STATUS_GEMV_HANDLE status;


    ///------balanced balanced2------///
    GEMV_INT_TYPE nthreads;
    GEMV_INT_TYPE* csrSplitter;
    GEMV_INT_TYPE* Yid;
    GEMV_INT_TYPE* Apinter;
    GEMV_INT_TYPE* Start1;
    GEMV_INT_TYPE* End1;
    GEMV_INT_TYPE* Start2;
    GEMV_INT_TYPE* End2;
    GEMV_INT_TYPE* Bpinter;
    ///------balanced balanced2------///


    ///---------sell C Sigma---------///

    ///---------sell C Sigma---------///
}gemv_Handle;

typedef gemv_Handle*  gemv_Handle_t;

float hsum_s_avx(__m256 in256) ;

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

#endif