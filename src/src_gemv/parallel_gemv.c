
#include <gemv.h>
void parallel_gemv(GEMV_INT_TYPE m,
                   const GEMV_INT_TYPE*RowPtr,
                   const GEMV_INT_TYPE *ColIdx,
                   const GEMV_VAL_TYPE*Matrix_Val,
                   const GEMV_VAL_TYPE*Vector_Val_X,
                   GEMV_VAL_TYPE*Vector_Val_Y){
#pragma omp parallel default(shared)
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = RowPtr[i] ; j < RowPtr[i + 1] ; j++) {
            sum += Matrix_Val[j] * Vector_Val_X[ColIdx[j]];
        }
        Vector_Val_Y[i] = sum;
    }
}