//
// Created by kouushou on 2020/12/8.
//

#ifndef GEMV_DOTPRODUCT_H
#define GEMV_DOTPRODUCT_H
#include "spmv_Defines.h"
float basic_s_dotProduct(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const float *Val, const float *X);

float basic_s_dotProduct_avx2(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const float *Val, const float *X);

float basic_s_dotProduct_avx512(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const float *Val, const float *X);


double basic_d_dotProduct(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const double *Val, const double *X) ;

double basic_d_dotProduct_avx2(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const double *Val, const double *X) ;

double basic_d_dotProduct_avx512(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE* indx, const double *Val, const double *X);

void Dot_Product_s_Selected(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE*indx, const void *Val, const void *X,
        void * res, VECTORIZED_WAY vectorizedWay);

void Dot_Product_d_Selected(
        BASIC_INT_TYPE len, const BASIC_INT_TYPE*indx, const void *Val, const void *X,
        void * res, VECTORIZED_WAY vectorizedWay);



typedef double (*dot_d_product_function)(BASIC_INT_TYPE len, const BASIC_INT_TYPE *indx,
                                         const double *Val, const double *X);

typedef float (*dot_s_product_function)(BASIC_INT_TYPE len, const BASIC_INT_TYPE *indx,
                                        const float *Val, const float *X);

typedef void (*dot_product_function)(BASIC_INT_TYPE len, const BASIC_INT_TYPE *indx,
                                     const void *Val, const void *X, void *res, VECTORIZED_WAY vec);

dot_product_function inner_basic_GetDotProduct(BASIC_SIZE_TYPE types);


#endif //GEMV_DOTPRODUCT_H
