//
// Created by kouushou on 2020/12/8.
//
#include <spmv.h>

#ifdef DOT_AVX2_CAN
float hsum_s_avx(__m256 in256) {
    float sum;

    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sum, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );

    return sum;
}

double hsum_d_avx(__m256d in256d){
    /*
    double sum = 0;
    double * s = (double *)&in256;
    for(int i = 0 ; i < 4 ; ++i)sum+=s[i];*/
    double sum;

    __m256d hsum = _mm256_add_pd(in256d, _mm256_permute2f128_pd(in256d, in256d, 0x1));
    _mm_store_sd(&sum, _mm_hadd_pd( _mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum) ) );

    return sum;
}

#endif



float basic_s_dotProduct(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const float *Val, const float *X) {
    float ret = 0;
    for(int i = 0 ; i < len ; ++i){
        ret+=Val[i]*X[indx[i]];
    }
    return ret;
}

float basic_s_dotProduct_avx2(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const float *Val, const float *X) {
#ifdef DOT_AVX2_CAN
    if(len >> 5) {
        float sum = 0;
        __m256 res = _mm256_setzero_ps();
        const int DEPTH = 8;
        int dif = len;
        int nloop = dif / DEPTH;
        int remainder = dif % DEPTH;
        for (int li = 0, j = 0; li < nloop; li++, j += DEPTH) {

            //__m256 vecv = _mm256_loadu_ps(&Val[j]);
            //__m256i veci = _mm256_loadu_si256((__m256i *) (&indx[j]));
            //__m256 vecx = _mm256_i32gather_ps(X, *(__m256i_u*)(indx+j), sizeof(X[0]));
            res = _mm256_fmadd_ps(*(__m256_u *) (Val + j),
                                  _mm256_i32gather_ps(X, *(__m256i_u *) (indx + j), sizeof(X[0])),
                                  res);
        }
        //Y[u] += _mm256_reduce_add_ps(res);
        sum += hsum_s_avx(res);

        for (int j = len - remainder; j < len; ++j) {
            sum += Val[j] * X[indx[j]];
        }
        return sum;
    }else{
        return basic_s_dotProduct(len, indx, Val, X);
    }
#else
    return basic_s_dotProduct(len, indx, Val, X);
#endif
}


float basic_s_dotProduct_avx512(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const float *Val, const float *X){
#ifdef DOT_AVX512_CAN
    float sum = 0;
    __m512 res = _mm512_setzero_ps();
    int dif = len;
    const int DEPTH=16;
    int nloop = dif / DEPTH;
    int remainder = dif % DEPTH;
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH)
    {
        //__m512 vecv = _mm512_loadu_ps(&Val[j]);
        //__m512i veci =  _mm512_loadu_si512(&indx[j]);
        //__m512 vecx = _mm512_i32gather_ps (_mm512_loadu_si512( (__m512i_u *)(indx+j)), X, sizeof(X[0]));
        res = _mm512_fmadd_ps(*(__m512_u*)(Val+j),
                              _mm512_i32gather_ps (_mm512_loadu_si512( (__m512i_u *)(indx+j)), X, sizeof(X[0])),
                              res);
    }
    sum += _mm512_reduce_add_ps(res);

    for (int j = nloop * DEPTH; j < len; j++) {
        sum += Val[j] * X[indx[j]];
    }
    return sum;
#elif DOT_AVX2_CAN
    return basic_s_dotProduct_avx2(len, indx, Val, X);
#else
    return basic_s_dotProduct(len, indx, Val, X);
#endif
}

double basic_d_dotProduct(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const double *Val, const double *X) {
    double ret = 0;
    for(int i = 0 ; i < len ; ++i){
        ret+=Val[i]*X[indx[i]];
    }
    return ret;
}

double basic_d_dotProduct_avx2(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const double *Val, const double *X) {
#ifdef DOT_AVX2_CAN
    if(len >> 4) {/// if len > 16
        double sum = 0;
        __m256d res = _mm256_setzero_pd();
        const int DEPTH = 4;
        int dif = len;
        int nloop = dif / DEPTH;
        int remainder = dif % DEPTH;
        for (int li = 0, j = 0; li < nloop; li++, j += DEPTH) {
            //__m256d vecx = _mm256_i32gather_pd(X,_mm256_castsi256_si128(*(__m256i_u*)(indx+j)),sizeof(X[0]));
            res = _mm256_fmadd_pd(
                    *((__m256d_u *) (Val + j)),
                    _mm256_i32gather_pd(X, _mm256_castsi256_si128(*(__m256i_u *) (indx + j)), sizeof(X[0])),
                    res);
        }
        //Y[u] += _mm256_reduce_add_ps(res);
        sum += hsum_d_avx(res);

        for (int j = len - remainder; j < len; ++j) {
            sum += Val[j] * X[indx[j]];
        }
        return sum;
    }else{
        return basic_d_dotProduct(len, indx, Val, X);
    }
#else
    return basic_d_dotProduct(len, indx, Val, X);
#endif
}


double basic_d_dotProduct_avx512(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const double *Val, const double *X){
#ifdef DOT_AVX512_CAN
    double sum = 0;
    __m512d res = _mm512_setzero_pd();
    int dif = len;
    const int DEPTH=8;
    int nloop = dif / DEPTH;
    int remainder = dif % DEPTH;
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH) {
        //__m512d vecv =*((__m512d_u *)(Val+j));// _mm512_loadu_pd(&Val[j]);
        //__m256i veci =  _mm256_loadu_si256((__m256i *) (&indx[j]));
        //__m512d vecx = _mm512_i32gather_pd (_mm256_loadu_si256((__m256i_u *) (indx+j)), X, sizeof(X[0]));
        /*
        res = _mm512_fmadd_pd(
                ,//_mm512_loadu_pd((__m512d_u *) (Val + j)),
                _mm512_i32gather_pd (_mm256_loadu_si256((__m256i_u *) (indx + j)), X, sizeof(X[0])),
                res);
                */
        res = _mm512_fmadd_pd(*((__m512d_u *)(Val+j)),
                              _mm512_i32gather_pd (_mm256_loadu_si256((__m256i_u *) (indx+j)), X, sizeof(X[0]))
                              ,res);
    }
    sum += _mm512_reduce_add_pd(res);

    for (int j = nloop * DEPTH; j < len; j++) {
        sum += Val[j] * X[indx[j]];
    }
    return sum;
#elif DOT_AVX2_CAN
    return basic_d_dotProduct_avx2(len,indx,Val,X);
#else
    return basic_d_dotProduct(len, indx, Val, X);
#endif
}


float (* const Dot_s_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE*
        indx, const float *Val, const float *X)={
                basic_s_dotProduct,
                basic_s_dotProduct_avx2,
                basic_s_dotProduct_avx512,
        };


double (* const Dot_d_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE*
        indx, const double *Val, const double *X)={
                basic_d_dotProduct,
                basic_d_dotProduct_avx2,
                basic_d_dotProduct_avx512,
        };



void Dot_Product_s_Selected(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE*indx, const void *Val, const void *X,
        void * res, VECTORIZED_WAY vectorizedWay
                          ){

    CONVERT_FLOAT(res) = Dot_s_Products[vectorizedWay](
            len,indx,CONVERT_FLOAT_T(Val),CONVERT_FLOAT_T(X));

}
void Dot_Product_d_Selected(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE*indx, const void *Val, const void *X,
        void * res, VECTORIZED_WAY vectorizedWay){
    CONVERT_DOUBLE(res) = Dot_d_Products[vectorizedWay](
            len,indx,CONVERT_DOUBLE_T(Val),CONVERT_DOUBLE_T(X));

}
void Dot_Product_d_aocl(BASIC_INT_TYPE len, const BASIC_INT_TYPE*indx, const void *Val, const void *X,
                        void * res, VECTORIZED_WAY vectorizedWay){
    __m256d vec_vals , vec_x , vec_y;
    const BASIC_INT_TYPE *colIndPtr;
    const double *matValPtr;
    const double * x = X;
    matValPtr = Val;
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
        vec_vals = _mm256_loadu_pd((double const *)matValPtr);

        //Gather the x vector elements from the column indices
        vec_x  =
                ///*
                 _mm256_set_pd(x[*(colIndPtr+3)],
                               x[*(colIndPtr+2)],
                               x[*(colIndPtr+1)],
                               x[*(colIndPtr)]);
        //*/

        //_mm256_i32gather_pd(x, _mm256_castsi256_si128(*(__m256i_u *) (colIndPtr)), sizeof(x[0]));
        vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

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

dot_product_function inner_basic_GetDotProduct(BASIC_SIZE_TYPE types){
    switch (types) {
        case sizeof(double ):{
            return Dot_Product_d_aocl;
        }break;
        case sizeof(float ):{
            return Dot_Product_s_Selected;
        }break;
        default:{
            return NULL;
        }break;
    }
}


