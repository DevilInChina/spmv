//
// Created by kouushou on 2020/12/6.
//
#include <spmv.h>
#define LINE_S_PRODUCT_PARAMETERS_IN const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,float *Vector_Y
#define LINE_D_PRODUCT_PARAMETERS_IN const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,double *Vector_Y
#define LINE_PRODUCT_PARAMETERS_CALL(banner) Val+(banner),indx+(banner),Vector_X,Vector_Y+(banner)

void basic_s_lineProduct_4(const float*Val, const BASIC_INT_TYPE* indx, const float *Vector_X, float *Vector_Y){
    Vector_Y[0] += Val[0]*Vector_X[indx[0]];
    Vector_Y[1] += Val[1]*Vector_X[indx[1]];
    Vector_Y[2] += Val[2]*Vector_X[indx[2]];
    Vector_Y[3] += Val[3]*Vector_X[indx[3]];
}
void basic_s_lineProduct_8(const float*Val, const BASIC_INT_TYPE* indx, const float *Vector_X, float *Vector_Y){
    for(unsigned int i = 0 ; i < 4 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

void basic_s_lineProduct_16(const float*Val, const BASIC_INT_TYPE* indx, const float *Vector_X, float *Vector_Y){
    for(unsigned int i = 0 ; i < 8 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

void basic_s_lineProduct_4_avx2(const float*Val, const BASIC_INT_TYPE* indx, const float *Vector_X, float *Vector_Y){
    basic_s_lineProduct_4(Val,indx,Vector_X,Vector_Y);
}

void basic_s_lineProduct_8_avx2(const float*Val, const BASIC_INT_TYPE* indx, const float *Vector_X, float *Vector_Y){
#ifdef DOT_AVX2_CAN

    __m256 vecv = _mm256_loadu_ps(&Val[0]);

    __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[0]));
    __m256 vecx = _mm256_i32gather_ps(Vector_X, veci, sizeof(Vector_X[0]));

    __m256 vecY = _mm256_loadu_ps(&Vector_Y[0]);

    vecY = _mm256_fmadd_ps(vecv,vecx,vecY);

    _mm256_store_ps(Vector_Y,vecY);

#else
    basic_s_lineProduct_8(Val,indx,Vector_X,Vector_Y);
#endif
}

void basic_s_lineProduct_16_avx2(LINE_S_PRODUCT_PARAMETERS_IN){
#ifdef DOT_AVX2_CAN
    for(int i = 0 ; i < 2 ; ++i){
        basic_s_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
#else
    basic_s_lineProduct_16(Val,indx,Vector_X,Vector_Y);
#endif
}
void basic_s_lineProduct_4_avx512(LINE_S_PRODUCT_PARAMETERS_IN) {
    basic_s_lineProduct_4(LINE_PRODUCT_PARAMETERS_CALL(0));
}

void basic_s_lineProduct_8_avx512(LINE_S_PRODUCT_PARAMETERS_IN) {
    basic_s_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(0));
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

void basic_d_lineProduct_4(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
    Vector_Y[0] += Val[0]*Vector_X[indx[0]];
    Vector_Y[1] += Val[1]*Vector_X[indx[1]];
    Vector_Y[2] += Val[2]*Vector_X[indx[2]];
    Vector_Y[3] += Val[3]*Vector_X[indx[3]];
}

void basic_d_lineProduct_8(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
    for(unsigned int i = 0 ; i < 4 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

void basic_d_lineProduct_16(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
    for(unsigned int i = 0 ; i < 8 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

#include <string.h>
void basic_d_lineProduct_4_avx2(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
#ifdef DOT_AVX2_CAN
    __m256d vecv = _mm256_load_pd(&Val[0]);

    __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[0]));
    __m128i vec128i = _mm256_castsi256_si128(veci);
    __m256d vecx = _mm256_i32gather_pd(Vector_X, vec128i, sizeof(Vector_X[0]));

    __m256d vecY = _mm256_load_pd(&Vector_Y[0]);

    vecY = _mm256_fmadd_pd(vecv,vecx,vecY);

    _mm256_store_pd(Vector_Y,vecY);

#else
    basic_d_lineProduct_4(Val,indx,Vector_X,Vector_Y);
#endif
}
void basic_d_lineProduct_8_avx2(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
#ifdef DOT_AVX2_CAN
    for(int i = 0 ; i < 2 ; ++i){
        basic_d_lineProduct_4_avx2(Val+i*4,indx+i*4,Vector_X,Vector_Y+i*4);
    }
#else
    basic_d_lineProduct_8(Val,indx,Vector_X,Vector_Y);
#endif
}

void basic_d_lineProduct_16_avx2(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
#ifdef DOT_AVX2_CAN
    for(int i = 0 ; i < 2 ; ++i){
        basic_d_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
#else
    basic_d_lineProduct_16(Val,indx,Vector_X,Vector_Y);
#endif
}


void basic_d_lineProduct_4_avx512(const double*Val, const BASIC_INT_TYPE* indx, const double *Vector_X, double *Vector_Y){
    basic_d_lineProduct_4_avx2(Val,indx,Vector_X,Vector_Y);
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
    basic_d_lineProduct_8(Val,indx,Vector_X,Vector_Y);
#endif
}

void basic_d_lineProduct_16_avx512(LINE_D_PRODUCT_PARAMETERS_IN){
    for(int i = 0 ; i < 2 ; ++i){
        basic_d_lineProduct_8_avx512(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
}

void (* const Line_s_Products[])
        (const float*Val,const BASIC_INT_TYPE* indx,
         const float *Vector_X,float *Vector_Y)={
                basic_s_lineProduct_4,
                basic_s_lineProduct_4_avx2,
                basic_s_lineProduct_4_avx512,
                basic_s_lineProduct_8,
                basic_s_lineProduct_8_avx2,
                basic_s_lineProduct_8_avx512,
                basic_s_lineProduct_16,
                basic_s_lineProduct_16_avx2,
                basic_s_lineProduct_16_avx512,
        };

void (* const Line_d_Products[])
        (const double*Val,const BASIC_INT_TYPE* indx,
         const double *Vector_X,double *Vector_Y)={
                basic_d_lineProduct_4,
                basic_d_lineProduct_4_avx2,
                basic_d_lineProduct_4_avx512,
                basic_d_lineProduct_8,
                basic_d_lineProduct_8_avx2,
                basic_d_lineProduct_8_avx512,
                basic_d_lineProduct_16,
                basic_d_lineProduct_16_avx2,
                basic_d_lineProduct_16_avx512,
        };
const char*Line_s_Products_name[]={
        "basic_s_lineProduct_4",
        "basic_s_lineProduct_4_avx2",
        "basic_s_lineProduct_4_avx512",
        "basic_s_lineProduct_8",
        "basic_s_lineProduct_8_avx2",
        "basic_s_lineProduct_8_avx512",
        "basic_s_lineProduct_16",
        "basic_s_lineProduct_16_avx2",
        "basic_s_lineProduct_16_avx512"
};
const char*Line_d_Products_name[]={
        "basic_d_lineProduct_4",
        "basic_d_lineProduct_4_avx2",
        "basic_d_lineProduct_4_avx512",
        "basic_d_lineProduct_8",
        "basic_d_lineProduct_8_avx2",
        "basic_d_lineProduct_8_avx512",
        "basic_d_lineProduct_16",
        "basic_d_lineProduct_16_avx2",
        "basic_d_lineProduct_16_avx512",
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
    int Level = 2;
    if(Level>2)Level = 2;
    if(Level<0)Level = 0;

    int Pack = pow_ksm(2,Level+2);

    BASIC_INT_TYPE HEX_turn = length / Pack;
    int i = 0;

    while (Pack>=4) {
        int functionChoose = Level*VECTOR_TOTAL_SIZE + vectorizedWay;
        for (; i+Pack < length; i += Pack) {
            Line_d_Products[functionChoose](Val+i,indx+i,Vector_X,Vector_Y+i);
        }
        Pack>>=1;
        --Level;
    }
    for(; i < length ; ++i){
        Vector_Y[i] += Val[i]*Vector_X[indx[i]];
    }
}

void basic_s_lineProduct(BASIC_INT_TYPE length, const float *Val, const BASIC_INT_TYPE* indx,
                         const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay){
    int Level = 2;

    if(Level>2)Level = 2;
    if(Level<0)Level = 0;

    int Pack = pow_ksm(2,Level+2);

    BASIC_INT_TYPE HEX_turn = length / Pack;
    int i = 0;

    while (Pack>=4) {
        int functionChoose = Level*VECTOR_TOTAL_SIZE+dotProductWay;
        for (; i+Pack < length; i += Pack) {
            Line_s_Products[functionChoose](Val+i,indx+i,Vector_X,Vector_Y+i);
        }
        Pack>>=1;
        --Level;
    }
    for(; i < length ; ++i){
        Vector_Y[i] += Val[i]*Vector_X[indx[i]];
    }
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
