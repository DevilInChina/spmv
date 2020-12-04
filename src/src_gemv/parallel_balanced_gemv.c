//
// Created by kouushou on 2020/11/25.
//

#include <gemv.h>

void parallel_balanced_gemv(
                            const gemv_Handle_t handle,
                            GEMV_INT_TYPE m,
                            const GEMV_INT_TYPE* RowPtr,
                            const GEMV_INT_TYPE* ColIdx,
                            const GEMV_VAL_TYPE* Matrix_Val,
                            const GEMV_VAL_TYPE* Vector_Val_X,
                            GEMV_VAL_TYPE*       Vector_Val_Y) {
    parallel_balanced_gemv_Selected
    (handle,m,RowPtr,ColIdx,Matrix_Val,
     Vector_Val_X,Vector_Val_Y,DOT_NONE);
}

void parallel_balanced_gemv_avx2(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y) {
    parallel_balanced_gemv_Selected
    (handle,m,RowPtr,ColIdx,Matrix_Val,
     Vector_Val_X,Vector_Val_Y,DOT_AVX2);
}


void parallel_balanced_gemv_avx512(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y) {
    parallel_balanced_gemv_Selected
    (handle,m,RowPtr,ColIdx,Matrix_Val,
     Vector_Val_X,Vector_Val_Y,DOT_AVX512);
}