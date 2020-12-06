//
// Created by kouushou on 2020/12/4.
//
#include <gemv.h>
#include <math.h>
#include <string.h>
void sell_C_Sigma_get_handle(gemv_Handle_t handle,
                             GEMV_INT_TYPE S,GEMV_INT_TYPE C,
                             GEMV_INT_TYPE m,
                             const GEMV_INT_TYPE*RowPtr,
                             GEMV_INT_TYPE nnzR,
                             GEMV_INT_TYPE nthreads){
    handle->S = S;
    handle->C = C;
    handle->Blos = floor((float)m / (float)S);

}
void sell_C_Sigma_gemv(const gemv_Handle_t handle,
                       GEMV_INT_TYPE m,
                       const GEMV_INT_TYPE* RowPtr,
                       const GEMV_INT_TYPE* ColIdx,
                       const GEMV_VAL_TYPE* Matrix_Val,
                       const GEMV_VAL_TYPE* Vector_Val_X,
                       GEMV_VAL_TYPE*       Vector_Val_Y){
/*
    int BloS = handle->Blos;
    int S = handle->S;
    int C = handle->C;
    int *Cmax = handle->Cmax;
    float *Y_first = (float *)malloc(sizeof(float) * (BloS * S));
    memset (Y_first, 0, sizeof(float) * (BloS * S));
    memset (Y_first, 0, sizeof(float) * (BloS * S));
    int redBloS = m - BloS * S;
    int loca = 0;
    for(int k = 0;k < S/C*BloS;k++)
    {
        if(k !=0 )
        {
            loca += Cmax[k-1];
        }
        for(int i = 0;i < C;i++)
        {
            float sum = 0;
            for(int p = 0;p < Cmax[k];p++)
            {
                sum += ValAT[C*loca+p*C+i] * X[ColAT[C*loca+p*C+i]];
            }
            Y_first[k*C+i] += sum;
        }
    }
    for(int i = 0;i < BloS * S;++i)
    {
        Vector_Val_Y[RowSort[i]] = Y_first[i];
    }
    if(redBloS != 0)
    {
        for (int i = 0; i < nnzred; i++)
        {
            float sum = 0;
            sum += Valred[i] * X[Colred[i]];
            Vector_Val_Y[Rowred[i]] += sum;
        }
    }
*/
}