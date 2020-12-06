
#include <gemv.h>
void parallel_gemv(GEMV_INT_TYPE m,
                   const GEMV_INT_TYPE*RowPtr,
                   const GEMV_INT_TYPE *ColIdx,
                   const GEMV_VAL_TYPE*Matrix_Val,
                   const GEMV_VAL_TYPE*Vector_Val_X,
                   GEMV_VAL_TYPE*Vector_Val_Y){
    GEMV_VAL_TYPE (*dot_product)(GEMV_INT_TYPE len, const GEMV_INT_TYPE *indx, const GEMV_VAL_TYPE *Val, const GEMV_VAL_TYPE *X)=
    inner__gemv_GetDotProduct(sizeof(GEMV_VAL_TYPE),DOT_NONE);
#pragma omp parallel default(shared)
    for (int i = 0; i < m; i++) {
        Vector_Val_Y[i] =
                dot_product(RowPtr[i+1]-RowPtr[i],
                                  ColIdx+RowPtr[i],Matrix_Val+RowPtr[i],Vector_Val_X);
    }
}