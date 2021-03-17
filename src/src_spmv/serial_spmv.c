//
// Created by kouushou on 2020/11/25.
//


#include "inner_spmv.h"


 void spmv_serial_cpp_d(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE *RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const double *Matrix_Val,
                            const double *Vector_Val_X,
                            double *Vector_Val_Y) {
    for (int i = 0; i < m; i++) {
        Dot_Product_d(RowPtr[i + 1] - RowPtr[i],
                           ColIdx + RowPtr[i],
                           Matrix_Val + RowPtr[i],
                           Vector_Val_X, Vector_Val_Y + i);
    }
}

 void spmv_serial_cpp_s(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE *RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const float *Matrix_Val,
                            const float *Vector_Val_X,
                            float *Vector_Val_Y) {
    for (int i = 0; i < m; i++) {
        Dot_Product_s(RowPtr[i + 1] - RowPtr[i],
                           ColIdx + RowPtr[i],
                           Matrix_Val + RowPtr[i],
                           Vector_Val_X, Vector_Val_Y + i);
    }
}

void spmv_serial_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y
) {
    if (handle->data_size == sizeof(double)) {
        spmv_serial_cpp_d(handle, m, RowPtr, ColIdx, (double *) Matrix_Val, (double *) Vector_Val_X,
                        (double *) Vector_Val_Y);
    } else {
        spmv_serial_cpp_s(handle, m, RowPtr, ColIdx, (float *) Matrix_Val, (float *) Vector_Val_X,
                        (float *) Vector_Val_Y);
    }
}


