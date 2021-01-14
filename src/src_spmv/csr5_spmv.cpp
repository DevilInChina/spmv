#include <algorithm>
#include "inner_spmv.h"
#include "csr5_avx2/anonymouslib_avx2.h"

void csr5HandleDestory(spmv_Handle_t handle){
    if(handle){
        if(handle->cppHandle){
            anonymouslibHandle<int,  int, double>&A = *( (anonymouslibHandle<int,  int, double>*)handle->cppHandle );
            A.destroy();
            free(handle->cppHandle);
            handle->cppHandle = nullptr;
        }
    }
}

void csr5Spmv_get_handle_Selected(spmv_Handle_t handle,
                                      BASIC_INT_TYPE m,
                                      BASIC_INT_TYPE n,
                                      BASIC_INT_TYPE*RowPtr,
                                      BASIC_INT_TYPE*ColIdx,
                                      const void*Matrix_Val
) {

    //printf("begin handle\n");
    handle->cppHandle = malloc(sizeof (anonymouslibHandle<int, int, double> ));
    *( (anonymouslibHandle<int,  int, double>*)handle->cppHandle ) = anonymouslibHandle<int,  int, double>(m,n);
    anonymouslibHandle<int,  int, double>&A = *( (anonymouslibHandle<int,  int, double>*)handle->cppHandle );

    int sigma = ANONYMOUSLIB_CSR5_SIGMA; //nnzA/(8*ANONYMOUSLIB_CSR5_OMEGA);
    A.inputCSR(RowPtr[m]-RowPtr[0], RowPtr, ColIdx, (double*)Matrix_Val);
   // A.setX(x);

    A.setSigma(sigma);
    A.asCSR5();
    //printf("handle done\n");
}

void spmv_csr5Spmv_Selected(const spmv_Handle_t handle,
                                BASIC_INT_TYPE m,
                                const BASIC_INT_TYPE* RowPtr,
                                const BASIC_INT_TYPE* ColIdx,
                                const void* Matrix_Val,
                                const void* Vector_Val_X,
                                void*       Vector_Val_Y
){
    anonymouslibHandle<int,  int, double>&A = *( (anonymouslibHandle<int,  int, double>*)handle->cppHandle );

    A.setX((double *)Vector_Val_X);
    memset(Vector_Val_Y,0,sizeof (double )*m);
    A.spmv(1,(double *)Vector_Val_Y);
}

int lower_bound(const int *first,const int *last,int key){
    return (int)(std::lower_bound(first,last,key) - first);
}

int upper_bound(const int *first,const int *last,int key){
    return (int)(std::upper_bound (first,last,key) - first);
}



