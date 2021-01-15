//
// Created by kouushou on 2021/1/15.
//
#include <spmv.h>
#define LINE_S_PRODUCTGather_PARAMETERS_IN const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y
#define LINE_D_PRODUCTGather_PARAMETERS_IN const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y
#define LINE_PRODUCTGather_PARAMETERS_CALL(banner) Val+(banner),indx+(banner),Vector_X,indy+(banner),Vector_Y



void basic_s_lineProductGather_len(LINE_S_PRODUCTGather_PARAMETERS_IN) {
    for (int i = 0; i < length; ++i) {
        Vector_Y[indy[i]] += Val[i] * Vector_X[indx[i]];
    }
}


void  basic_d_lineProductGather_len(LINE_D_PRODUCTGather_PARAMETERS_IN) {
    for (int i = 0; i < length; ++i) {
        Vector_Y[indy[i]] += Val[i] * Vector_X[indx[i]];
    }
}
void basic_s_lineProductGather_avx2 (const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y){
#ifdef DOT_AVX2_CAN
    const int block = 8;
    int remain = length % block;
    int len = length - remain;

    for(int i = 0 ; i < len ; i+=block){
        
    }
    if(remain){
        basic_s_lineProductGather_len(remain,LINE_PRODUCTGather_PARAMETERS_CALL(len));
    }
#else
    basic_s_lineProductGather_len(length,LINE_PRODUCTGather_PARAMETERS_CALL(0));
#endif
}
void basic_d_lineProductGather_avx2(const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y);

void basic_s_lineProductGather_avx512 (const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y);

void basic_d_lineProductGather_avx512(const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y);


line_product_gather_function inner_basic_GetLineProductGather(BASIC_SIZE_TYPE types, VECTORIZED_WAY vec) {

    switch (types) {
        case sizeof(double): {
            switch (vec) {
                default: {
                    return (line_product_gather_function) basic_d_lineProductGather_len;
                }
            }
        }
        default: {
            switch (vec) {
                default: {
                    return (line_product_gather_function) basic_s_lineProductGather_len;
                }
            }
        }
    }
}
