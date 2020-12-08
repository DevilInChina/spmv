//
// Created by kouushou on 2020/12/6.
//

#ifndef GEMV_LINEPRODUCT_H
#define GEMV_LINEPRODUCT_H
#include "gemv_Defines.h"

extern
void (* const Line_s_Products[9])
        (const float*Val,const GEMV_INT_TYPE* indx,
         const float *Vector_X,float *Vector_Y);

extern
void (* const Line_d_Products[9])
        (const double*Val,const GEMV_INT_TYPE* indx,
         const double *Vector_X,double *Vector_Y);

extern const char*Line_s_Products_name[9];
extern const char*Line_d_Products_name[9];

void gemv_d_lineProduct(GEMV_INT_TYPE length, const double*Val, const GEMV_INT_TYPE* indx,
                        const double *Vector_X, double *Vector_Y, VECTORIZED_WAY vectorizedWay);

void gemv_s_lineProduct(GEMV_INT_TYPE length, const float *Val, const GEMV_INT_TYPE* indx,
                        const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay);

void gemv_d_lineProduct_set_zero(GEMV_INT_TYPE length, const double*Val, const GEMV_INT_TYPE* indx,
                                 const double *Vector_X, double *Vector_Y, VECTORIZED_WAY dotProductWay);

void gemv_s_lineProduct_set_zero(GEMV_INT_TYPE length, const float *Val, const GEMV_INT_TYPE* indx,
                                 const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay);

void gemv_s_gather(GEMV_INT_TYPE length,const float *Val,const GEMV_INT_TYPE*indx,float *Vector_Y);

void gemv_d_gather(GEMV_INT_TYPE length,const double *Val,const GEMV_INT_TYPE*indx,double *Vector_Y);
#endif //GEMV_LINEPRODUCT_H
