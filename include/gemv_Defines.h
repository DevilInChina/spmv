//
// Created by kouushou on 2020/12/6.
//

#ifndef GEMV_GEMV_DEFINES_H
#define GEMV_GEMV_DEFINES_H

#ifndef BASIC_INT_TYPE
#define BASIC_INT_TYPE int
#endif

#ifndef BASIC_VAL_TYPE
#define BASIC_VAL_TYPE float
#endif

#ifndef BASIC_SIZE_TYPE
#define BASIC_SIZE_TYPE unsigned long
#endif

typedef enum VECTORIZED_WAY{
    VECTOR_NONE,
    VECTOR_AVX2,
    VECTOR_AVX512,
    VECTOR_TOTAL_SIZE
}VECTORIZED_WAY;


typedef enum STATUS_GEMV_HANDLE{
    STATUS_NONE,
    STATUS_BALANCED,
    STATUS_BALANCED2,
    STATUS_SELL_C_SIGMA,
    STATUS_TOTAL_SIZE
}STATUS_GEMV_HANDLE;

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


typedef struct gemv_Handle {
    STATUS_GEMV_HANDLE status;


    ///------balanced balanced2------///
    BASIC_INT_TYPE nthreads;
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
}gemv_Handle;

typedef gemv_Handle*  gemv_Handle_t;

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

#define GET_GEMV(FUNCTION,VECTORIZED) ((FUNCTION-1)*(VECTOR_AVX512+1) + (VECTORIZED))

extern void (* const gemv[])
        (const gemv_Handle_t handle,
         BASIC_INT_TYPE m,
         const BASIC_INT_TYPE* RowPtr,
         const BASIC_INT_TYPE* ColIdx,
         const BASIC_VAL_TYPE* Matrix_Val,
         const BASIC_VAL_TYPE* Vector_Val_X,
         BASIC_VAL_TYPE*       Vector_Val_Y);
extern const char*gemv_name[];


#define CONVERT_FLOAT(pointer) *((float*)(pointer))
#define CONVERT_DOUBLE(pointer) *((double*)(pointer))
#define CONVERT_FLOAT_T(pointer) ((float*)(pointer))
#define CONVERT_DOUBLE_T(pointer) ((double*)(pointer))

#define CONVERT_EQU(pointer,size,other) ((size)==sizeof(double))?(CONVERT_DOUBLE(pointer)=(other)):(CONVERT_FLOAT(pointer)=(other))
#define CONVERT_ADDEQU(pointer1,size,pointer2) \
((size)==sizeof(double))?                      \
(CONVERT_DOUBLE(pointer1)+=CONVERT_DOUBLE(pointer2)):(CONVERT_FLOAT(pointer1)+=CONVERT_FLOAT(pointer2))


#define PARAMETER_HANDLE_IN(type) \
const gemv_Handle_t handle , \
BASIC_INT_TYPE m,\
const BASIC_INT_TYPE*RowPtr,\
const BASIC_INT_TYPE *ColIdx,\
const type *Matrix_Val,\
const type *Vector_Val_X, \
type *Vector_Val_Y

#define PARAMETER_NO_HANDLE_IN(type) BASIC_INT_TYPE m,\
const BASIC_INT_TYPE*RowPtr,\
const BASIC_INT_TYPE *ColIdx,\
const type *Matrix_Val,\
const type *Vector_Val_X, \
type *Vector_Val_Y

#define PARAMETER_HANDLE_CALL handle,m, RowPtr, ColIdx, Matrix_Val, Vector_Val_X, Vector_Val_Y
#define PARAMETER_NO_HANDLE_CALL m, RowPtr, ColIdx, Matrix_Val, Vector_Val_X, Vector_Val_Y

#define FUNC_HANDLE_DECLARES(ret_type,header,type,VECTOR_CHOICE) \
ret_type spmv_##header##_##type##_##VECTOR_CHOICE(PARAMETER_HANDLE_IN(type))

#define FUNC_NO_HANDLE_DECLARES(ret_type,header,type,VECTOR_CHOICE) \
ret_type spmv_##header##_##type##_##VECTOR_CHOICE(PARAMETER_NO_HANDLE_IN(type))

#define FUNC_HANDLE_DEFINES(ret_type,header,type,VECTOR_CHOICE) \
ret_type spmv_##header##_##type##_##VECTOR_CHOICE(PARAMETER_HANDLE_IN(type)) {  \
    spmv_##header##_Selected(PARAMETER_HANDLE_CALL,sizeof(type),VECTOR_CHOICE);\
}

#define FUNC_NO_HANDLE_DEFINES(ret_type,header,type,VECTOR_CHOICE) \
ret_type spmv_##header##_##type##_##VECTOR_CHOICE(PARAMETER_NO_HANDLE_IN(type)) {  \
    spmv_##header##_Selected(PARAMETER_NO_HANDLE_CALL,sizeof(type),VECTOR_CHOICE);\
}

#define FUNC_DECLARES_TYPE(DEC,name,VECTOR_CHOICE) \
DEC(void,name,double,VECTOR_CHOICE);\
DEC(void,name,float,VECTOR_CHOICE)\

#define FUNC_DECLARES(DEC,name) \
FUNC_DECLARES_TYPE(DEC,name,VECTOR_NONE); \
FUNC_DECLARES_TYPE(DEC,name,VECTOR_AVX2); \
FUNC_DECLARES_TYPE(DEC,name,VECTOR_AVX512)

extern float (* const Dot_s_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE*
        indx, const float *Val, const float *X);


extern double (* const Dot_d_Products[])
        (BASIC_INT_TYPE len, const BASIC_INT_TYPE*
        indx, const double *Val, const double *X);
#endif //GEMV_GEMV_DEFINES_H
