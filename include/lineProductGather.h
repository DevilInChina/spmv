//
// Created by kouushou on 2021/1/15.
//

#ifndef SPMV_LINEPRODUCTGATHER_H
#define SPMV_LINEPRODUCTGATHER_H

#endif //SPMV_LINEPRODUCTGATHER_H

typedef void (*line_s_product_gather_function) (const BASIC_INT_TYPE ld, const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y);

typedef void (*line_d_product_gather_function) (const BASIC_INT_TYPE ld,const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y);

typedef void (*line_product_gather_function) (const BASIC_INT_TYPE ld,const BASIC_INT_TYPE length, const void*Val,const BASIC_INT_TYPE* indx, const void *Vector_X,const BASIC_INT_TYPE*indy,void *Vector_Y);


void basic_s_lineProductGather_len (const BASIC_INT_TYPE ld,const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y);

void basic_d_lineProductGather_len(const BASIC_INT_TYPE ld,const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y);

void basic_s_lineProductGather_avx2 (const BASIC_INT_TYPE ld,const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y);

void basic_d_lineProductGather_avx2(const BASIC_INT_TYPE ld,const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y);


line_product_gather_function inner_basic_GetLineProductGather(BASIC_SIZE_TYPE types,VECTORIZED_WAY vec);