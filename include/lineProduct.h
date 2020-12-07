//
// Created by kouushou on 2020/12/6.
//

#ifndef GEMV_LINEPRODUCT_H
#define GEMV_LINEPRODUCT_H
#include "gemv_Defines.h"

void gemv_s_lineProduct_4(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_8(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_16(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_4_avx2(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_8_avx2(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_16_avx2(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_4_avx512(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y) ;

void gemv_s_lineProduct_8_avx512(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_s_lineProduct_16_avx512(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y);

void gemv_d_lineProduct_4(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_8(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_16(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_4_avx2(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_8_avx2(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_16_avx2(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_4_avx512(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_8_avx512(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

void gemv_d_lineProduct_16_avx512(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y);

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
#endif //GEMV_LINEPRODUCT_H
