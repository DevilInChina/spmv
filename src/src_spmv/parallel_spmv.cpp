

#include "inner_spmv.h"
inline void Dot_Product_d_aocl(BASIC_INT_TYPE len, const BASIC_INT_TYPE*indx, const void *Val, const void *X,
                               void * res, VECTORIZED_WAY vectorizedWay){
    __m256d vec_vals , vec_x , vec_y;
    const BASIC_INT_TYPE *colIndPtr;
    const double *matValPtr;
    const double * x = (double *)X;
    matValPtr = (double *)Val;
    colIndPtr = indx;
    BASIC_INT_TYPE j;
    double result = 0.0;
    vec_y = _mm256_setzero_pd();
    BASIC_INT_TYPE nnzThisLine = len;
    BASIC_INT_TYPE k_iter = nnzThisLine / 4;
    BASIC_INT_TYPE k_rem = nnzThisLine % 4;

    //Loop in multiples of 4 non-zeroes
    for(j =  0 ; j < k_iter ; j++ )
    {
        //(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
        //vec_vals = _mm256_loadu_pd((double const *)matValPtr);

        //Gather the x vector elements from the column indices
        //vec_x  =
        /*
         _mm256_set_pd(x[*(colIndPtr+3)],
                       x[*(colIndPtr+2)],
                       x[*(colIndPtr+1)],
                       x[*(colIndPtr)]);
//*/

        //_mm256_i32gather_pd(x, _mm256_castsi256_si128(*(__m256i_u *) (colIndPtr)), sizeof(x[0]));
        vec_y = _mm256_fmadd_pd(
                *((__m256d_u *) (matValPtr)),
                _mm256_set_pd(x[*(colIndPtr+3)],
                              x[*(colIndPtr+2)],
                              x[*(colIndPtr+1)],
                              x[*(colIndPtr)]),
                vec_y);

        matValPtr+=4;
        colIndPtr+=4;
    }

    // Horizontal addition
    if(k_iter){
        // sum[0] += sum[1] ; sum[2] += sum[3]
        vec_y = _mm256_hadd_pd(vec_y, vec_y);
        // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
        __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
        // Extract 128 bits to obtain sum[2] and sum[3]
        __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
        // Add remaining two sums
        __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
        // Store result
        result = sse_sum[0];
    }

    //Remainder loop for nnzThisLine%4
    for(j =  0 ; j < k_rem ; j++ )
    {
        result += *matValPtr++ * x[*colIndPtr++];
    }



    *(double *)res = result ;
}
void spmv_parallel_Selected(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE*RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const void *Matrix_Val,
                            const void *Vector_Val_X,
                            void *Vector_Val_Y
                   ) {

    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY vectorizedWay = handle->vectorizedWay;
    dot_product_function dotProductFunction = inner_basic_GetDotProduct(size);
    const double *Val = (double *)Matrix_Val;
    double *Y = (double *)Vector_Val_Y;
    const double * x = (double *)Vector_Val_X;
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        Dot_Product_d_aocl(RowPtr[i + 1] - RowPtr[i],
                           ColIdx + RowPtr[i],
                           Val + RowPtr[i] ,
                           x,
                           Y + i ,
                           vectorizedWay);
    }
}
