//
// Created by kouushou on 2020/12/6.
//
#include <spmv.h>
#include <string.h>
#define LINE_S_PRODUCT_PARAMETERS_IN const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,float *Vector_Y
#define LINE_D_PRODUCT_PARAMETERS_IN const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,double *Vector_Y
#define LINE_PRODUCT_PARAMETERS_CALL(banner) Val+(banner),indx+(banner),Vector_X,Vector_Y+(banner)



 void basic_s_lineProduct_len(int len ,LINE_S_PRODUCT_PARAMETERS_IN){
    for(int i = 0 ; i < len ; ++ i){
        Vector_Y[i]+=Val[i]*Vector_X[indx[i]];
    }
}


 void basic_d_lineProduct_len(int len ,LINE_D_PRODUCT_PARAMETERS_IN){
    for(int i = 0 ; i < len ; ++ i){
        Vector_Y[i]+=Val[i]*Vector_X[indx[i]];
    }
}

void basic_s_lineProduct_8_avx2(const float*Val, const BASIC_INT_TYPE* indx, const float *Vector_X, float *Vector_Y){
#ifdef DOT_AVX2_CAN


    //__m256 vecx = _mm256_i32gather_ps(Vector_X, *((__m256i*)indx), sizeof(Vector_X[0]));

    *(__m256_u *) (Vector_Y ) = _mm256_fmadd_ps(*(__m256_u *)(Val),
                                                _mm256_i32gather_ps(Vector_X, *((__m256i*)indx), sizeof(Vector_X[0])),
                                                *(__m256_u *) (Vector_Y ));

#else
    basic_s_lineProduct_len(8,LINE_PRODUCT_PARAMETERS_CALL(0));
#endif
}


void basic_s_lineProduct_16_avx512(LINE_S_PRODUCT_PARAMETERS_IN){
#ifdef DOT_AVX512_CAN
    __m512 vecv = _mm512_loadu_ps(&Val[0]);

    __m512i veci =  _mm512_loadu_si512(&indx[0]);
    __m512 vecx = _mm512_i32gather_ps (veci, Vector_X, sizeof(Vector_X[0]));


    __m512 vecY = _mm512_loadu_ps(&Vector_Y[0]);

    vecY = _mm512_fmadd_ps(vecv,vecx,vecY);

    _mm512_store_ps(Vector_Y,vecY);
#else
    for(int i = 0 ; i < 2 ; ++i){
        basic_s_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
#endif
}

typedef void (*line_d_function)(const double*Val, const BASIC_INT_TYPE* indx,
                                        const double *Vector_X, double *Vector_Y);

typedef void (*line_s_function)(const float *Val, const BASIC_INT_TYPE* indx,
                                        const float *Vector_X, float *Vector_Y);

void basic_d_lineProduct_4_avx2(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
#ifdef DOT_AVX2_CAN
      //__m256d vecv = _mm256_loadu_pd(&Val[0]);

       //__m256i veci = _mm256_loadu_si256((__m256i_u*)indx);
       //__m128i vec128i = _mm256_castsi256_si128(*(__m256i_u *) (indx));
       //__m256d vecx =_mm256_i32gather_pd(Vector_X, _mm256_castsi256_si128(*(__m256i_u *) (indx)), sizeof(Vector_X[0]));

       //__m256d vecY = _mm256_load_pd(&Vector_Y[0]);
        *(__m256d_u *) (Vector_Y )  = _mm256_fmadd_pd(
               *(__m256d_u*)(Val),
               _mm256_i32gather_pd(Vector_X, _mm256_castsi256_si128(*(__m256i_u *) (indx)), sizeof(Vector_X[0])),
               *(__m256d_u *) (Vector_Y ));

       //_mm256_store_pd(Vector_Y,vecY);
#else
    basic_d_lineProduct_len(4,LINE_PRODUCT_PARAMETERS_CALL(0));
#endif
}

void basic_d_lineProduct_8_avx512(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y) {
#ifdef DOT_AVX512_CAN
    __m512d vecv = _mm512_loadu_pd(&Val[0]);

    __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[0]));
    __m512d vecx = _mm512_i32gather_pd (veci, Vector_X, sizeof(Vector_X[0]));

    __m512d vecY = _mm512_loadu_pd(&Vector_Y[0]);
    vecY = _mm512_fmadd_pd(vecv,vecx,vecY);

    _mm512_store_pd(Vector_Y, vecY);

#else
    for(int i = 0 ; i < 2; ++i) {
        basic_d_lineProduct_4_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*4));
    }
#endif
}

line_s_function func_cal_s_8_16[]={
        basic_s_lineProduct_8_avx2,
        basic_s_lineProduct_16_avx512
};

line_d_function func_cal_d_4_8[]={
        basic_d_lineProduct_4_avx2,
        basic_d_lineProduct_8_avx512
};

int pow2(unsigned int l){
    int odd = 1;
    unsigned int a=2;
    while (l){
        if(l&1)odd*=a;
        a*=a;
        l>>=1;
    }
    return odd;
}
int pow_ksm(unsigned int a,unsigned int  b) {
    int odd, y;
    odd = 1;
    y = a;
    while (b) {
        if (b & 1) odd *= y;
        y *= y;
        b >>= 1;
    }
    return odd;
}

void basic_d_lineProduct(BASIC_INT_TYPE length, const double*Val, const BASIC_INT_TYPE* indx,
                         const double *Vector_X, double *Vector_Y, VECTORIZED_WAY vectorizedWay){
    int caller = (int)vectorizedWay-VECTOR_AVX2;
    if(caller<0 || caller > 1){
        basic_d_lineProduct_len(length,LINE_PRODUCT_PARAMETERS_CALL(0));
        return;
    }
    const int block = 2 << vectorizedWay;
    int i;
    for(i = 0 ; i + block  < length ; i+=block){
        func_cal_d_4_8[caller](LINE_PRODUCT_PARAMETERS_CALL(i));
    }

    basic_d_lineProduct_len(length-i, LINE_PRODUCT_PARAMETERS_CALL(i));
}

void basic_s_lineProduct(BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                         const float *Vector_X, float *Vector_Y, VECTORIZED_WAY vectorizedWay){

    int caller = (int)vectorizedWay-VECTOR_AVX2;

    if(caller<0 || caller > 1){
        basic_s_lineProduct_len(length,LINE_PRODUCT_PARAMETERS_CALL(0));
        return;
    }

    const int block = 4 << vectorizedWay;
    int i;
    for( i = 0 ; i + block  < length ; i+=block){
        func_cal_s_8_16[caller](LINE_PRODUCT_PARAMETERS_CALL(i));
    }


    basic_s_lineProduct_len(length-i, LINE_PRODUCT_PARAMETERS_CALL(i));
}

void basic_d_lineProduct_set_zero(BASIC_INT_TYPE length, const double*Val, const BASIC_INT_TYPE* indx,
                                  const double *Vector_X, double *Vector_Y, VECTORIZED_WAY dotProductWay){
    memset(Vector_Y,0,sizeof(double )*length);
    basic_d_lineProduct(length, Val, indx, Vector_X, Vector_Y, dotProductWay);
}

void basic_s_lineProduct_set_zero(BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                                  const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay){
    memset(Vector_Y,0,sizeof(float )*length);
    basic_s_lineProduct(length, Val, indx, Vector_X, Vector_Y, dotProductWay);
}

void Line_Product_s_Selected(
        BASIC_INT_TYPE length, const void*Val, const BASIC_INT_TYPE* indx,
        const void *Vector_X, void *Vector_Y, VECTORIZED_WAY dotProductWay
){
    basic_s_lineProduct(length, CONVERT_FLOAT_T(Val), indx,
                        CONVERT_FLOAT_T(Vector_X), CONVERT_FLOAT_T(Vector_Y), dotProductWay);
}
void Line_Product_d_Selected(
        BASIC_INT_TYPE length, const void*Val, const BASIC_INT_TYPE* indx,
        const void *Vector_X, void *Vector_Y, VECTORIZED_WAY dotProductWay
){
    basic_d_lineProduct(length, CONVERT_DOUBLE_T(Val), indx,
                        CONVERT_DOUBLE_T(Vector_X), CONVERT_DOUBLE_T(Vector_Y), dotProductWay);
}


line_product_function inner_basic_GetLineProduct(BASIC_SIZE_TYPE types){
    switch (types) {
        case sizeof(double ):{
            return Line_Product_d_Selected;
        }break;
        case sizeof(float ):{
            return Line_Product_s_Selected;
        }break;
        default:{
            return NULL;
        }break;
    }
}

void basic_s_gather(BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE*indx, void *Vector_Y, VECTORIZED_WAY vec){
    for(int i = 0 ; i < length ; ++i){
        *(CONVERT_FLOAT_T(Vector_Y)+indx[i]) = *(CONVERT_FLOAT_T(Val)+i);
    }
}

void basic_d_gather(BASIC_INT_TYPE length, const void *Val, const BASIC_INT_TYPE*indx, void *Vector_Y, VECTORIZED_WAY vec){
    for(int i = 0 ; i < length ; ++i){
        *(CONVERT_DOUBLE_T(Vector_Y)+indx[i]) = *(CONVERT_DOUBLE_T(Val)+i);
    }
}
gather_function inner_basic_GetGather(BASIC_SIZE_TYPE types){
    switch (types) {
        case sizeof(double ):{
            return basic_d_gather;
        }break;
        case sizeof(float ):{
            return basic_s_gather;
        }break;
        default:{
            return NULL;
        }break;
    }
}



void basic_s_pack_lineProduct(BASIC_INT_TYPE pack_size,BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                              const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay) {
    memset(Vector_Y,0,sizeof (float )*length);
    for (int i = 0; i < pack_size; ++i) {
        basic_s_lineProduct(length,  LINE_PRODUCT_PARAMETERS_CALL(i*length)-i*length, dotProductWay);
    }
}


void basic_d_pack_lineProduct(BASIC_INT_TYPE pack_size,BASIC_INT_TYPE length, const double *Val, const BASIC_INT_TYPE* indx,
                              const double *Vector_X, double *Vector_Y, VECTORIZED_WAY dotProductWay) {
    memset(Vector_Y,0,sizeof (double )*length);
    for (int i = 0; i < pack_size; ++i) {
        basic_d_lineProduct(length, LINE_PRODUCT_PARAMETERS_CALL(i*length)-i*length, dotProductWay);
    }
}


void PackLine_Product_s_Selected(BASIC_INT_TYPE pack_size,
        BASIC_INT_TYPE length, const void*Val, const BASIC_INT_TYPE* indx,
        const void *Vector_X, void *Vector_Y, VECTORIZED_WAY dotProductWay
){
    basic_s_pack_lineProduct(pack_size,length, CONVERT_FLOAT_T(Val), indx,
                        CONVERT_FLOAT_T(Vector_X), CONVERT_FLOAT_T(Vector_Y), dotProductWay);
}
void PackLine_Product_d_Selected(BASIC_INT_TYPE pack_size,
        BASIC_INT_TYPE length, const void*Val, const BASIC_INT_TYPE* indx,
        const void *Vector_X, void *Vector_Y, VECTORIZED_WAY dotProductWay
){
    basic_d_pack_lineProduct(pack_size,length, CONVERT_DOUBLE_T(Val), indx,
                        CONVERT_DOUBLE_T(Vector_X), CONVERT_DOUBLE_T(Vector_Y), dotProductWay);
}

packLine_product_function inner_basic_GetPackLineProduct(BASIC_SIZE_TYPE types){
    switch (types) {
        case sizeof(double ):{
            return PackLine_Product_d_Selected;
        }break;
        case sizeof(float ):{
            return PackLine_Product_s_Selected;
        }break;
        default:{
            return NULL;
        }break;
    }
}
