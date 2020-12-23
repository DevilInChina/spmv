//
// Created by kouushou on 2020/12/6.
//

#ifndef GEMV_SPMV_DEFINES_H
#define GEMV_SPMV_DEFINES_H

#ifndef BASIC_INT_TYPE
#define BASIC_INT_TYPE int
#endif

#ifndef BASIC_SIZE_TYPE
#define BASIC_SIZE_TYPE unsigned long
#endif

typedef enum VECTORIZED_WAY{
    VECTOR_NONE,
    VECTOR_AVX2,
    VECTOR_AVX512,
    VECTOR_TOTAL_SIZE /// count total ways of vectorized
}VECTORIZED_WAY;
extern const char*Vectorized_names[];

typedef enum SPMV_METHODS{
    Method_Serial,
    Method_Parallel,
    Method_Balanced,
    Method_Balanced2,
    Method_SellCSigma,
    Method_CSR5SPMV,
    Method_Total_Size /// count total ways of methods
}SPMV_METHODS;
extern const char*Methods_names[];

extern const char * funcNames[];

typedef struct Row_Block {
    const BASIC_INT_TYPE   *indxBegin;
    const void   *valBegin;
    BASIC_INT_TYPE   length;
    BASIC_INT_TYPE   rowNumber;
}Row_Block,*Row_Block_t;

typedef struct C_Block{
    BASIC_INT_TYPE *ColIndex;
    BASIC_INT_TYPE ld;
    BASIC_INT_TYPE *RowIndex;
    BASIC_INT_TYPE C;
    void *ValT;
    void *Y;
}C_Block,*C_Block_t;


typedef struct spmv_Handle {
    SPMV_METHODS spmvMethod;
    BASIC_SIZE_TYPE data_size;
    BASIC_SIZE_TYPE nthreads;
    VECTORIZED_WAY vectorizedWay;


    ///------balanced balanced2------///
    BASIC_INT_TYPE* csrSplitter;
    BASIC_INT_TYPE* Yid;
    BASIC_INT_TYPE* Apinter;
    BASIC_INT_TYPE* Start1;
    BASIC_INT_TYPE* End1;
    BASIC_INT_TYPE* Start2;
    BASIC_INT_TYPE* End2;
    BASIC_INT_TYPE* Bpinter;
    ///------balanced balanced2------///


    ///---------sell C Sigma---------///
    BASIC_INT_TYPE Sigma,C;
    BASIC_INT_TYPE banner;
    C_Block_t C_Blocks;
    ///---------sell C Sigma---------///

    ///---------csr  5  spmv ---------///

    void *cppHandle;

}spmv_Handle;

typedef spmv_Handle*  spmv_Handle_t;

typedef enum LINE_PRODUCT_WAY{
    GEMV_LINE_PRODUCT_4,
    GEMV_LINE_PRODUCT_4_AVX2,
    GEMV_LINE_PRODUCT_4_AVX512,
    GEMV_LINE_PRODUCT_8,
    GEMV_LINE_PRODUCT_8_AVX2,
    GEMV_LINE_PRODUCT_8_AVX512,
    GEMV_LINE_PRODUCT_16,
    GEMV_LINE_PRODUCT_16_AVX2,
    GEMV_LINE_PRODUCT_16_AVX512,
}LINE_PRODUCT_WAY;

extern
void (* const Line_s_Products[])
        (const float*Val,const BASIC_INT_TYPE* indx,
         const float *Vector_X,float *Vector_Y);

extern
void (* const Line_d_Products[])
        (const double*Val,const BASIC_INT_TYPE* indx,
         const double *Vector_X,double *Vector_Y);

extern const char*Line_s_Products_name[];

extern const char*Line_d_Products_name[];



#define CONVERT_FLOAT(pointer) *((float*)(pointer))
#define CONVERT_DOUBLE(pointer) *((double*)(pointer))
#define CONVERT_FLOAT_T(pointer) ((float*)(pointer))
#define CONVERT_DOUBLE_T(pointer) ((double*)(pointer))

#define CONVERT_EQU(pointer,size,other) ((size)==sizeof(double))?(CONVERT_DOUBLE(pointer)=(other)):(CONVERT_FLOAT(pointer)=(other))
#define CONVERT_ADDEQU(pointer1,size,pointer2) \
((size)==sizeof(double))?                      \
(CONVERT_DOUBLE(pointer1)+=CONVERT_DOUBLE(pointer2)):(CONVERT_FLOAT(pointer1)+=CONVERT_FLOAT(pointer2))



extern float (* const Dot_s_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE*
        indx, const float *Val, const float *X);


extern double (* const Dot_d_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE*
        indx, const double *Val, const double *X);
#endif //GEMV_SPMV_DEFINES_H
