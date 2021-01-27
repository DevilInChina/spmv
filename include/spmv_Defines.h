//
// Created by kouushou on 2020/12/6.
//
#if defined(__cplusplus)
extern "C" {
#endif
#ifndef GEMV_SPMV_DEFINES_H
#define GEMV_SPMV_DEFINES_H

#ifndef BASIC_INT_TYPE
#define BASIC_INT_TYPE int
#endif

#ifndef BASIC_SIZE_TYPE
#define BASIC_SIZE_TYPE unsigned long
#endif

typedef enum VECTORIZED_WAY {
    VECTOR_NONE,
    VECTOR_AVX2,
    VECTOR_AVX512,
    VECTOR_TOTAL_SIZE /// count total ways of vectorized
} VECTORIZED_WAY;
extern const char *Vectorized_names[];

typedef enum SPMV_METHODS {
    Method_Serial,
    Method_Parallel,
    Method_Balanced,
    Method_Balanced2,
    Method_SellCSigma,
    Method_CSR5SPMV,
    Method_Total_Size,
    Method_Numa ///count total ways of methods
} SPMV_METHODS;
extern const char *Methods_names[];

extern const char *funcNames[];

typedef struct Row_Block {
    const BASIC_INT_TYPE *indxBegin;
    const void *valBegin;
    BASIC_INT_TYPE length;
    BASIC_INT_TYPE rowNumber;
} Row_Block, *Row_Block_t;


typedef struct Sigma_Block {
    BASIC_INT_TYPE C;
    BASIC_INT_TYPE times;
    BASIC_INT_TYPE *ld;
    BASIC_INT_TYPE *ColIndex;
    BASIC_INT_TYPE *RowIndex;
    BASIC_INT_TYPE total;
    void *ValT;
    //void *Y;
} Sigma_Block, *Sigma_Block_t;


typedef struct spmv_Handle {
    SPMV_METHODS spmvMethod;
    BASIC_SIZE_TYPE data_size;
    BASIC_SIZE_TYPE nthreads;
    VECTORIZED_WAY vectorizedWay;
    int Level_3_opt_used;
    BASIC_INT_TYPE *RowPtr;
    BASIC_INT_TYPE *ColIdx;
    BASIC_INT_TYPE *index;
    void *Matrix_Val;
    void *Y_temp;

    ///------balanced balanced2------///

    ///------balanced balanced2------///


    ///---------sell C Sigma---------///
    BASIC_INT_TYPE Sigma, C;
    BASIC_INT_TYPE banner;
    Sigma_Block_t sigmaBlock;
    ///---------sell C Sigma---------///

    ///---------csr  5  spmv ---------///

    void *extraHandle;

} spmv_Handle;

typedef spmv_Handle *spmv_Handle_t;


#define CONVERT_FLOAT(pointer) *((float*)(pointer))
#define CONVERT_DOUBLE(pointer) *((double*)(pointer))
#define CONVERT_FLOAT_T(pointer) ((float*)(pointer))
#define CONVERT_DOUBLE_T(pointer) ((double*)(pointer))

#define CONVERT_EQU(pointer, size, other) ((size)==sizeof(double))?(CONVERT_DOUBLE(pointer)=(other)):(CONVERT_FLOAT(pointer)=(other))
#define CONVERT_ADDEQU(pointer1, size, pointer2) \
((size)==sizeof(double))?                      \
(CONVERT_DOUBLE(pointer1)+=CONVERT_DOUBLE(pointer2)):(CONVERT_FLOAT(pointer1)+=CONVERT_FLOAT(pointer2))


extern float (*const Dot_s_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE *
        indx, const float *Val, const float *X);


extern double (*const Dot_d_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE *
        indx, const double *Val, const double *X);

#endif //GEMV_SPMV_DEFINES_H

#if defined(__cplusplus)
}
#endif