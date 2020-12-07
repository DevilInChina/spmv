//
// Created by kouushou on 2020/12/6.
//

#include <gemv.h>
#include <stdio.h>

void Test_s_LineProduct(const char*funcName,void (* LineProducts)
        (const float*Val,const GEMV_INT_TYPE* indx,
         const float *Vector_X,float *Vector_Y),int len,int iter){
    int lenC = len;
    //len<<=1;
    float *X = malloc(sizeof(float )*len);
    int *indx = malloc(sizeof(int )*len);
    float *Val = malloc(sizeof(float )*len);
    float *Y_golden = malloc(sizeof(float )*len);
    float *Y = malloc(sizeof(float )*len);
    for(int i = 0 ; i < lenC ; ++i){
        indx[i] = lenC-1-i;
        X[i] = i+1;
        Val[i]=i+1;
    }
    for(int i = 0 ; i < lenC ; ++i){
        Y_golden[i] = Val[i]*X[indx[i]];
    }
    for(int i = 0 ; i < iter ; ++i){
        LineProducts(Val,indx,X,Y);
    }
    int cnt = 0;
    for(int i = 0 ; i < lenC ; ++i){
        if(Y[i]!=Y_golden[i]){
            ++cnt;
            printf("%f %f\n",Y[i],Y_golden[i]);
        }
    }
    printf("%s_errcnt=  %d in %d\n",funcName,cnt,lenC);
    free(Y_golden);
    free(Val);
    free(Y);
    free(indx);
    free(X);

}
void Test_d_LineProduct(const char*funcName,void (* LineProducts)
        (const double*Val,const GEMV_INT_TYPE* indx,
         const double *Vector_X,double *Vector_Y),int len,int iter){

    int lenC = len;
    //len<<=1;
    double *X = aligned_alloc(1024,sizeof(double )*len);
    int *indx = aligned_alloc(1024,sizeof(int )*len);
    double *Val = aligned_alloc(1024,sizeof(double )*len);
    double *Y_golden = aligned_alloc(1024,sizeof(double )*len);
    double *Y = aligned_alloc(1024,sizeof(double )*len);
    for(int i = 0 ; i < lenC ; ++i){
        indx[i] = lenC-1-i;
        X[i] = i+1;
        Val[i]=i+1;
    }
    for(int i = 0 ; i < lenC ; ++i){
        Y_golden[i] = Val[i]*X[indx[i]];
    }
    for(int i = 0 ; i < iter ; ++i){
        LineProducts(Val,indx,X,Y);
    }
    int cnt = 0;
    for(int i = 0 ; i < lenC ; ++i){
        if(Y[i]!=Y_golden[i]){
            ++cnt;
            printf("%f %f\n",Y[i],Y_golden[i]);
        }
    }
    printf("%s_errcnt=  %d in %d\n",funcName,cnt,lenC);
    free(Y_golden);
    free(Val);
    free(Y);
    free(indx);
    free(X);

}
int main(){
    for(int i = 0 ; i < 9 ; ++i){
        Test_d_LineProduct(Line_d_Products_name[i],Line_d_Products[i],4<<(i/3),1 );
        Test_s_LineProduct(Line_s_Products_name[i],Line_s_Products[i],4<<(i/3),1 );
    }
}
