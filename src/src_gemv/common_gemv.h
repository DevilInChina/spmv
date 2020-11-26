//
// Created by kouushou on 2020/11/25.
//

#ifndef GEMV_COMMON_GEMV_H
#define GEMV_COMMON_GEMV_H
#include <gemv.h>

typedef enum DOT_PRODUCT_WAY{
    DOT_NONE,
    DOT_AVX2,
    DOT_AVX512
}DOT_PRODUCT_WAY;

struct gemv_Handle {
    STATUS_GEMV_HANDLE status;
    GEMV_INT_TYPE nthreads;
    GEMV_INT_TYPE* csrSplitter;
    GEMV_INT_TYPE* Yid;
    GEMV_INT_TYPE* Apinter;
    GEMV_INT_TYPE* Start1;
    GEMV_INT_TYPE* End1;
    GEMV_INT_TYPE* Start2;
    GEMV_INT_TYPE* End2;
    GEMV_INT_TYPE* Bpinter;
};

float hsum_avx(__m256 in256) ;

float gemv_s_dotProduct(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X);

float gemv_s_dotProduct_avx2(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X);

float gemv_s_dotProduct_avx512(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X);


#endif //GEMV_COMMON_GEMV_H
