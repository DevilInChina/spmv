//
// Created by kouushou on 2020/12/6.
//
#include <gemv.h>
#define LINE_S_PRODUCT_PARAMETERS_IN const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y
#define LINE_D_PRODUCT_PARAMETERS_IN const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y
#define LINE_PRODUCT_PARAMETERS_CALL(banner) Val+(banner),indx+(banner),Vector_X,Vector_Y+(banner)

void gemv_s_lineProduct_4(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y){
    Vector_Y[0] += Val[0]*Vector_X[indx[0]];
    Vector_Y[1] += Val[1]*Vector_X[indx[1]];
    Vector_Y[2] += Val[2]*Vector_X[indx[2]];
    Vector_Y[3] += Val[3]*Vector_X[indx[3]];
}
void gemv_s_lineProduct_8(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y){
    for(unsigned int i = 0 ; i < 4 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

void gemv_s_lineProduct_16(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y){
    for(unsigned int i = 0 ; i < 8 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

void gemv_s_lineProduct_4_avx2(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y){
    gemv_s_lineProduct_4(Val,indx,Vector_X,Vector_Y);
}

void gemv_s_lineProduct_8_avx2(const float*Val,const GEMV_INT_TYPE* indx, const float *Vector_X,float *Vector_Y){
#ifdef DOT_AVX2_CAN

    __m256 vecv = _mm256_loadu_ps(&Val[0]);

    __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[0]));
    __m256 vecx = _mm256_i32gather_ps(Vector_X, veci, sizeof(Vector_X[0]));

    __m256 vecY = _mm256_loadu_ps(&Vector_Y[0]);

    vecY = _mm256_fmadd_ps(vecv,vecx,vecY);

    _mm256_store_ps(Vector_Y,vecY);

#else
    gemv_s_lineProduct_8(Val,indx,Vector_X,Vector_Y);
#endif
}

void gemv_s_lineProduct_16_avx2(LINE_S_PRODUCT_PARAMETERS_IN){
#ifdef DOT_AVX2_CAN
    for(int i = 0 ; i < 2 ; ++i){
        gemv_s_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
#else
    gemv_s_lineProduct_16(Val,indx,Vector_X,Vector_Y);
#endif
}
void gemv_s_lineProduct_4_avx512(LINE_S_PRODUCT_PARAMETERS_IN) {
    gemv_s_lineProduct_4(LINE_PRODUCT_PARAMETERS_CALL(0));
}

void gemv_s_lineProduct_8_avx512(LINE_S_PRODUCT_PARAMETERS_IN) {
    gemv_s_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(0));
}
void gemv_s_lineProduct_16_avx512(LINE_S_PRODUCT_PARAMETERS_IN){
#ifdef DOT_AVX512_CAN
    __m512 vecv = _mm512_loadu_ps(&Val[0]);

    __m512i veci =  _mm512_loadu_si512(&indx[0]);
    __m512 vecx = _mm512_i32gather_ps (veci, Vector_X, sizeof(Vector_X[0]));


    __m512 vecY = _mm512_loadu_ps(&Vector_Y[0]);

    vecY = _mm512_fmadd_ps(vecv,vecx,vecY);

    _mm512_store_ps(Vector_Y,vecY);
#else
    for(int i = 0 ; i < 2 ; ++i){
        gemv_s_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
#endif
}

void gemv_d_lineProduct_4(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
    Vector_Y[0] += Val[0]*Vector_X[indx[0]];
    Vector_Y[1] += Val[1]*Vector_X[indx[1]];
    Vector_Y[2] += Val[2]*Vector_X[indx[2]];
    Vector_Y[3] += Val[3]*Vector_X[indx[3]];
}

void gemv_d_lineProduct_8(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
    for(unsigned int i = 0 ; i < 4 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

void gemv_d_lineProduct_16(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
    for(unsigned int i = 0 ; i < 8 ; ++i){
        unsigned int li = i<<1u;
        unsigned int ri = i<<1u|1u;
        Vector_Y[li] += Val[li]*Vector_X[indx[li]];
        Vector_Y[ri] += Val[ri]*Vector_X[indx[ri]];
    }
}

#include <string.h>
void gemv_d_lineProduct_4_avx2(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
#ifdef DOT_AVX2_CAN
    __m256d vecv = _mm256_load_pd(&Val[0]);

    __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[0]));
    __m128i vec128i = _mm256_castsi256_si128(veci);
    __m256d vecx = _mm256_i32gather_pd(Vector_X, vec128i, sizeof(Vector_X[0]));

    __m256d vecY = _mm256_load_pd(&Vector_Y[0]);

    vecY = _mm256_fmadd_pd(vecv,vecx,vecY);

    _mm256_store_pd(Vector_Y,vecY);

#else
    gemv_d_lineProduct_4(Val,indx,Vector_X,Vector_Y);
#endif
}
void gemv_d_lineProduct_8_avx2(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
#ifdef DOT_AVX2_CAN
    for(int i = 0 ; i < 2 ; ++i){
        gemv_d_lineProduct_4_avx2(Val+i*4,indx+i*4,Vector_X,Vector_Y+i*4);
    }
#else
    gemv_d_lineProduct_8(Val,indx,Vector_X,Vector_Y);
#endif
}

void gemv_d_lineProduct_16_avx2(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
#ifdef DOT_AVX2_CAN
    for(int i = 0 ; i < 2 ; ++i){
        gemv_d_lineProduct_8_avx2(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
#else
    gemv_d_lineProduct_16(Val,indx,Vector_X,Vector_Y);
#endif
}


void gemv_d_lineProduct_4_avx512(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y){
    gemv_d_lineProduct_4_avx2(Val,indx,Vector_X,Vector_Y);
}

void gemv_d_lineProduct_8_avx512(const double*Val,const GEMV_INT_TYPE* indx, const double *Vector_X,double *Vector_Y) {
#ifdef DOT_AVX512_CAN
    __m512d vecv = _mm512_loadu_pd(&Val[0]);

    __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[0]));
    __m512d vecx = _mm512_i32gather_pd (veci, Vector_X, sizeof(Vector_X[0]));

    __m512d vecY = _mm512_loadu_pd(&Vector_Y[0]);
    vecY = _mm512_fmadd_pd(vecv,vecx,vecY);

    _mm512_store_pd(Vector_Y, vecY);

#else
    gemv_d_lineProduct_8(Val,indx,Vector_X,Vector_Y);
#endif
}

void gemv_d_lineProduct_16_avx512(LINE_D_PRODUCT_PARAMETERS_IN){
    for(int i = 0 ; i < 2 ; ++i){
        gemv_d_lineProduct_8_avx512(LINE_PRODUCT_PARAMETERS_CALL(i*8));
    }
}

void (* const Line_s_Products[9])
        (const float*Val,const GEMV_INT_TYPE* indx,
         const float *Vector_X,float *Vector_Y)={
                gemv_s_lineProduct_4,
                gemv_s_lineProduct_4_avx2,
                gemv_s_lineProduct_4_avx512,
                gemv_s_lineProduct_8,
                gemv_s_lineProduct_8_avx2,
                gemv_s_lineProduct_8_avx512,
                gemv_s_lineProduct_16,
                gemv_s_lineProduct_16_avx2,
                gemv_s_lineProduct_16_avx512,
        };

void (* const Line_d_Products[9])
        (const double*Val,const GEMV_INT_TYPE* indx,
         const double *Vector_X,double *Vector_Y)={
                gemv_d_lineProduct_4,
                gemv_d_lineProduct_4_avx2,
                gemv_d_lineProduct_4_avx512,
                gemv_d_lineProduct_8,
                gemv_d_lineProduct_8_avx2,
                gemv_d_lineProduct_8_avx512,
                gemv_d_lineProduct_16,
                gemv_d_lineProduct_16_avx2,
                gemv_d_lineProduct_16_avx512,
        };
const char*Line_s_Products_name[9]={
        "gemv_s_lineProduct_4",
        "gemv_s_lineProduct_4_avx2",
        "gemv_s_lineProduct_4_avx512",
        "gemv_s_lineProduct_8",
        "gemv_s_lineProduct_8_avx2",
        "gemv_s_lineProduct_8_avx512",
        "gemv_s_lineProduct_16",
        "gemv_s_lineProduct_16_avx2",
        "gemv_s_lineProduct_16_avx512"
};
const char*Line_d_Products_name[9]={
        "gemv_d_lineProduct_4",
        "gemv_d_lineProduct_4_avx2",
        "gemv_d_lineProduct_4_avx512",
        "gemv_d_lineProduct_8",
        "gemv_d_lineProduct_8_avx2",
        "gemv_d_lineProduct_8_avx512",
        "gemv_d_lineProduct_16",
        "gemv_d_lineProduct_16_avx2",
        "gemv_d_lineProduct_16_avx512",
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

void gemv_d_lineProduct(GEMV_INT_TYPE length, const double*Val, const GEMV_INT_TYPE* indx,
                        const double *Vector_X, double *Vector_Y, VECTORIZED_WAY vectorizedWay){
    int Level = 2;
    if(Level>2)Level = 2;
    if(Level<0)Level = 0;

    int Pack = pow_ksm(2,Level+2);

    GEMV_INT_TYPE HEX_turn = length/Pack;
    int i = 0;

    while (Pack>=4) {
        int functionChoose = Level*3 + vectorizedWay;
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

void gemv_s_lineProduct(GEMV_INT_TYPE length, const float *Val, const GEMV_INT_TYPE* indx,
                        const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay){
    int Level = 2;

    if(Level>2)Level = 2;
    if(Level<0)Level = 0;

    int Pack = pow_ksm(2,Level+2);

    GEMV_INT_TYPE HEX_turn = length/Pack;
    int i = 0;

    while (Pack>=4) {
        int functionChoose = Level*3+dotProductWay;
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

void gemv_d_lineProduct_set_zero(GEMV_INT_TYPE length, const double*Val, const GEMV_INT_TYPE* indx,
                                 const double *Vector_X, double *Vector_Y, VECTORIZED_WAY dotProductWay){
    memset(Vector_Y,0,sizeof(double )*length);
    gemv_d_lineProduct(length,Val,indx,Vector_X,Vector_Y,dotProductWay);
}

void gemv_s_lineProduct_set_zero(GEMV_INT_TYPE length, const float *Val, const GEMV_INT_TYPE* indx,
                                 const float *Vector_X, float *Vector_Y, VECTORIZED_WAY dotProductWay){
    memset(Vector_Y,0,sizeof(float )*length);
    gemv_s_lineProduct(length,Val,indx,Vector_X,Vector_Y,dotProductWay);
}

void gemv_s_gather(GEMV_INT_TYPE length,const float *Val,const GEMV_INT_TYPE*indx,float *Vector_Y){
    for(int i = 0 ; i < length ; ++i){
        Vector_Y[indx[i]] = Val[i];
    }
}

void gemv_d_gather(GEMV_INT_TYPE length,const double *Val,const GEMV_INT_TYPE*indx,double *Vector_Y){
    for(int i = 0 ; i < length ; ++i){
        Vector_Y[indx[i]] = Val[i];
    }
}