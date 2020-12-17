//
// Created by kouushou on 2020/12/8.
//
#include <spmv.h>

#ifdef DOT_AVX2_CAN
float hsum_s_avx(__m256 in256) {
    float sum;

    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sum, _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));

    return sum;
}

double hsum_d_avx(__m256d in256){
    double sum;
    __m256d hsum = _mm256_hadd_pd(in256, in256);
    hsum = _mm256_add_pd(hsum, _mm256_permute2f128_pd(hsum, hsum, 0x1));
    _mm_store_sd(&sum, _mm_hadd_pd(_mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum)));

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
    float sum = 0;
    __m256 res = _mm256_setzero_ps();
    const int DEPTH = 8;
    int dif = len;
    int nloop = dif / DEPTH;
    int remainder = dif % DEPTH;
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH) {

        __m256 vecv = _mm256_loadu_ps(&Val[j]);
        __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[j]));
        __m256 vecx = _mm256_i32gather_ps(X, veci, sizeof(X[0]));
        res = _mm256_fmadd_ps(vecv, vecx, res);
    }
    //Y[u] += _mm256_reduce_add_ps(res);
    sum += hsum_s_avx(res);

    for (int j = nloop * DEPTH; j < len; ++j) {
        sum += Val[j] * X[indx[j]];
    }
    return sum;
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
        __m512 vecv = _mm512_loadu_ps(&Val[j]);
        __m512i veci =  _mm512_loadu_si512(&indx[j]);
        __m512 vecx = _mm512_i32gather_ps (veci, X, sizeof(X[0]));
        res = _mm512_fmadd_ps(vecv, vecx, res);
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
    double sum = 0;
    __m256d res = _mm256_setzero_pd();
    const int DEPTH = 4;
    int dif = len;
    int nloop = dif / DEPTH;
    int remainder = dif % DEPTH;
    long long high[2]={0,0};
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH) {

        __m256d vecv = _mm256_load_pd(&Val[j]);
        __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[j]));
        __m128i vec128i = _mm256_castsi256_si128(veci);
        __m256d vecx = _mm256_i32gather_pd(X, vec128i, sizeof(X[0]));
        res = _mm256_fmadd_pd(vecv, vecx, res);
    }
    //Y[u] += _mm256_reduce_add_ps(res);
    sum += hsum_d_avx(res);

    for (int j = nloop * DEPTH; j < len; ++j) {
        sum += Val[j] * X[indx[j]];
    }
    return sum;
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
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH)
    {
        __m512d vecv = _mm512_loadu_pd(&Val[j]);
        __m256i veci =  _mm256_loadu_si256((__m256i *) (&indx[j]));
        __m512d vecx = _mm512_i32gather_pd (veci, X, sizeof(X[0]));
        res = _mm512_fmadd_pd(vecv, vecx, res);
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


dot_product_function inner_basic_GetDotProduct(BASIC_SIZE_TYPE types){
    switch (types) {
        case sizeof(double ):{
            return Dot_Product_d_Selected;
        }break;
        case sizeof(float ):{
            return Dot_Product_s_Selected;
        }break;
        default:{
            return NULL;
        }break;
    }
}


