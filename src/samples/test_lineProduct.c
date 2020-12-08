//
// Created by kouushou on 2020/12/6.
//

#include <gemv.h>
#include <stdio.h>
#include <string.h>
#define ALIEN 32

void Test_s_LineProduct(const char*funcName,void (* LineProducts)
        (const float*Val,const BASIC_INT_TYPE* indx,
         const float *Vector_X,float *Vector_Y),int len,int iter){
    int lenC = len;
    //len<<=1;
    float *X = aligned_alloc(ALIEN,sizeof(float )*len);
    int *indx = aligned_alloc(ALIEN,sizeof(int )*len);
    float *Val = aligned_alloc(ALIEN,sizeof(float )*len);
    float *Y_golden = aligned_alloc(ALIEN,sizeof(float )*len);
    float *Y = aligned_alloc(ALIEN,sizeof(float )*len);
    for(int i = 0 ; i < lenC ; ++i){
        indx[i] = lenC-1-i;
        X[i] = i+1;
        Val[i]=i+1;
    }
    for(int i = 0 ; i < lenC ; ++i){
        Y_golden[i] = Val[i]*X[indx[i]];
    }
    for(int i = 0 ; i < iter ; ++i){
        memset(Y,0,sizeof(float )*lenC);
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
        (const double*Val,const BASIC_INT_TYPE* indx,
         const double *Vector_X,double *Vector_Y),int len,int iter){

    int lenC = len;
    //len<<=1;

    double *X = aligned_alloc(ALIEN,sizeof(double )*len);

    int *indx = aligned_alloc(ALIEN,sizeof(int )*len);

    double *Val = aligned_alloc(ALIEN,sizeof(double )*len);
    double *Y_golden = aligned_alloc(ALIEN,sizeof(double )*len);
    double *Y = aligned_alloc(ALIEN,sizeof(double )*len);
    for(int i = 0 ; i < lenC ; ++i){
        indx[i] = lenC-1-i;
        X[i] = i+1;
        Val[i]=i+1;
    }
    for(int i = 0 ; i < lenC ; ++i){
        Y_golden[i] = Val[i]*X[indx[i]];
    }
    for(int i = 0 ; i < iter ; ++i){
        memset(Y,0,sizeof(double )*lenC);
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
