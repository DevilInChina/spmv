//
// Created by kouushou on 2020/12/6.
//

#include <gemv.h>
#include <stdio.h>

void TestLineProduct(void (* LineProducts)
        (const GEMV_VAL_TYPE*Val,const GEMV_INT_TYPE* indx,
         const GEMV_VAL_TYPE *Vector_X,GEMV_VAL_TYPE *Vector_Y),int len,int iter){

    double *X = malloc(sizeof(double )*len);
    int *indx = malloc(sizeof(int )*len);
    double *Val = malloc(sizeof(double )*len);
    double *Y_golden = malloc(sizeof(double )*len);
    double *Y = malloc(sizeof(double )*len);
    for(int i = 0 ; i < len ; ++i){
        indx[i] = len-1-i;
        X[i] = 0.1*i+0.1;
        Val[i]=0.1*i+0.1;
    }
    for(int i = 0 ; i < len ; ++i){
        Y_golden[i] = Val[i]*X[indx[i]];
    }
    for(int i = 0 ; i < iter ; ++i){
        LineProducts(Val,indx,X,Y);
    }
    int cnt = 0;
    for(int i = 0 ; i < len ; ++i){
        if(Y[i]!=Y_golden[i]){
            ++cnt;
            printf("%f %f\n",Y[i],Y_golden[i]);
        }
    }
    printf("%d\n",cnt);
    free(Y_golden);
    free(Val);
    free(Y);
    free(indx);
    free(X);

}

int main(){
    TestLineProduct(gemv_d_lineProduct_4_avx2,4,10);
}