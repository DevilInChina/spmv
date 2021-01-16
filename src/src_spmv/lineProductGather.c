//
// Created by kouushou on 2021/1/15.
//
#include <spmv.h>

#define LINE_S_PRODUCTGather_PARAMETERS_IN const BASIC_INT_TYPE ld, const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y
#define LINE_D_PRODUCTGather_PARAMETERS_IN const BASIC_INT_TYPE ld, const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y
#define LINE_PRODUCTGather_PARAMETERS_CALL(banner) Val+(banner),indx+(banner),Vector_X,indy+(banner),Vector_Y



void basic_s_lineProductGather_len(LINE_S_PRODUCTGather_PARAMETERS_IN) {
    for(int i = 0 ; i < ld ; ++i){
        for(int j = 0 ; j < length ; ++j){
            Vector_Y[indy[j]]+=Val[i*length+j] * Vector_X[indx[i*length+j]];
        }
    }
}


void  basic_d_lineProductGather_len(LINE_D_PRODUCTGather_PARAMETERS_IN) {
    for(int i = 0 ; i < ld ; ++i){
        for(int j = 0 ; j < length ; ++j){
            Vector_Y[indy[j]]+=Val[i*length+j] * Vector_X[indx[i*length+j]];
        }
    }
}
void basic_s_lineProductGather_avx2 (LINE_S_PRODUCTGather_PARAMETERS_IN){
#ifdef DOT_AVX2_CAN
    const int block = 8;
    for(int i = 0 ; i < length ; i+=block){
        __m256_u vecy = _mm256_setzero_ps();
        for(int j = 0 ; j < ld ; ++j){
            vecy = _mm256_fmadd_ps(
                    *(__m256_u *) (Val + j*length+i),
                    _mm256_i32gather_ps(Vector_X, *(__m256i_u*)(indx+j*length+i),
                                        sizeof(Vector_X[0])),vecy
            );
        }
        float *cur = (float*)(&vecy);
        for(int j = 0 ; j < block ; ++j){
            Vector_Y[indy[i+j]] = cur[j];
        }
    }
#else
    basic_s_lineProductGather_len(length,LINE_PRODUCTGather_PARAMETERS_CALL(0));
#endif
}
void basic_d_lineProductGather_avx2(LINE_D_PRODUCTGather_PARAMETERS_IN){
#ifdef DOT_AVX2_CAN
    const int block = 4;
    for(int i = 0 ; i < length ; i+=block){
        __m256d_u vecy = _mm256_setzero_pd();
        for(int j = 0 ; j < ld ; ++j){
            vecy = _mm256_fmadd_pd(
                    *(__m256d_u *) (Val + j*length+i),
                    _mm256_i32gather_pd(Vector_X, _mm256_castsi256_si128(*(__m256i_u *) (indx + i + j*length)),
                                        sizeof(Vector_X[0])),vecy
                    );
        }
        double *cur = (double*)(&vecy);
        for(int j = 0 ; j < block ; ++j){
            Vector_Y[indy[i+j]] = cur[j];
        }
    }
#else
#endif
}



line_product_gather_function inner_basic_GetLineProductGather(BASIC_SIZE_TYPE types, VECTORIZED_WAY vec) {

    switch (types) {
        case sizeof(double): {
            switch (vec) {
                case VECTOR_NONE:{
                    return (line_product_gather_function) basic_d_lineProductGather_len;
                }
                default: {
                    return (line_product_gather_function) basic_d_lineProductGather_avx2;
                }
            }
        }
        default: {
            switch (vec) {
                case VECTOR_NONE:{
                    return (line_product_gather_function) basic_s_lineProductGather_len;
                }
                default: {
                    return (line_product_gather_function) basic_s_lineProductGather_avx2;
                }
            }
        }
    }
}
