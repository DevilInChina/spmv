//
// Created by kouushou on 2020/11/25.
//


#include <gemv.h>
void serial_gemv(GEMV_INT_TYPE m,
                 const GEMV_INT_TYPE*RowPtr,
                 const GEMV_INT_TYPE *ColIdx,
                 const GEMV_VAL_TYPE*Matrix_Val,
                 const GEMV_VAL_TYPE*Vector_Val_X,
                 GEMV_VAL_TYPE*Vector_Val_Y) {
    for (int i = 0; i < m; i++) {
        Vector_Val_Y[i] =
                gemv_s_dotProduct(RowPtr[i+1]-RowPtr[i],
                                  ColIdx+RowPtr[i],Matrix_Val+RowPtr[i],Vector_Val_X);
    }
}