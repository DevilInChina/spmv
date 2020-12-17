//
// Created by kouushou on 2020/12/6.
//

#ifndef GEMV_LINEPRODUCT_H
#define GEMV_LINEPRODUCT_H
#include "spmv_Defines.h"


void basic_d_lineProduct(BASIC_INT_TYPE length, const double*Val, const BASIC_INT_TYPE* indx,
                         const double *Vector_X, double *Vector_Y, VECTORIZED_WAY vectorizedWay);

void basic_s_lineProduct(BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                         const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay);

void basic_d_lineProduct_set_zero(BASIC_INT_TYPE length, const double*Val, const BASIC_INT_TYPE* indx,
                                  const double *Vector_X, double *Vector_Y, VECTORIZED_WAY dotProductWay);


void basic_s_lineProduct_set_zero(BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                                  const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay);

void basic_s_gather(BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE*indx, void *Vector_Y, VECTORIZED_WAY vec);

void basic_d_gather(BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE*indx, void *Vector_Y, VECTORIZED_WAY vec);


void basic_s_pack_lineProduct(BASIC_INT_TYPE pack_size,BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                         const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay);

void basic_d_pack_lineProduct(BASIC_INT_TYPE pack_size,BASIC_INT_TYPE length, const double *Val, const BASIC_INT_TYPE* indx,
                              const double *Vector_X, double *Vector_Y, VECTORIZED_WAY dotProductWay);



typedef void (*gather_function)(BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE*indx,
                                void *Vector_Y, VECTORIZED_WAY vec);

typedef void (*line_d_product_function)(BASIC_INT_TYPE length, const double*Val, const BASIC_INT_TYPE* indx,
                                        const double *Vector_X, double *Vector_Y);

typedef void (*line_s_product_function)(BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                                        const float *Vector_X, float *Vector_Y);

typedef void (*line_product_function)(BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE* indx,
                                      const void *Vector_X, void *Vector_Y, VECTORIZED_WAY vec);

typedef void (*packLine_product_function)(BASIC_INT_TYPE pack,BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE* indx,
                                      const void *Vector_X, void *Vector_Y, VECTORIZED_WAY vec);

line_product_function inner_basic_GetLineProduct(BASIC_SIZE_TYPE types);

gather_function inner_basic_GetGather(BASIC_SIZE_TYPE types);

packLine_product_function inner_basic_GetPackLineProduct(BASIC_SIZE_TYPE types);
#endif //GEMV_LINEPRODUCT_H
