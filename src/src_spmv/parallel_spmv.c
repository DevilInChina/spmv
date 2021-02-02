

#include "inner_spmv.h"
void spmv_parallel_Selected(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE*RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const void *Matrix_Val,
                            const void *Vector_Val_X,
                            void *Vector_Val_Y
                   ) {

    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY vectorizedWay = handle->vectorizedWay;
    dot_product_function dotProductFunction = inner_basic_GetDotProduct(size);
    const double *Val = Matrix_Val;
    double *Y = Vector_Val_Y;
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        dotProductFunction(RowPtr[i + 1] - RowPtr[i],
                           ColIdx + RowPtr[i],
                           Val + RowPtr[i] ,
                           Vector_Val_X,
                           Y + i ,
                           vectorizedWay);
    }
}
