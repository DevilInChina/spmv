//
// Created by kouushou on 2020/11/25.
//


#include "inner_spmv.h"

void spmv_serial_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void*Matrix_Val,
        const void*Vector_Val_X,
        void*Vector_Val_Y
                 ) {
    dot_product_function dot_product = inner_basic_GetDotProduct(handle->data_size);

    for (int i = 0; i < m; i++) {
        dot_product(RowPtr[i+1]-RowPtr[i],
                    ColIdx+RowPtr[i],Matrix_Val+RowPtr[i]*handle->data_size, Vector_Val_X,Vector_Val_Y+i*handle->data_size,
                    handle->vectorizedWay);

    }
}


