//
// Created by kouushou on 2020/12/6.
//

#include <spmv.h>
#include <stdio.h>
#include <string.h>
#define ALIEN 32
#define Test_LineProduct(type)\
void Test_##type##_LineProduct(const char*funcName,void (* LineProducts)\
    (const type*Val,const BASIC_INT_TYPE* indx,\
    const type *Vector_X,type *Vector_Y),int len,int iter) {\
    int lenC = len;           \
    type *X = aligned_alloc(ALIEN, sizeof(type) * len);\
    int *indx = aligned_alloc(ALIEN, sizeof(int) * len);\
    type *Val = aligned_alloc(ALIEN, sizeof(type) * len);\
    type *Y_golden = aligned_alloc(ALIEN, sizeof(type) * len);\
    type *Y = aligned_alloc(ALIEN, sizeof(type) * len);\
    for (int i = 0; i < lenC; ++i) {\
        indx[i] = lenC - 1 - i;\
        X[i] = i + 1;\
        Val[i] = i + 1;\
    }\
    for (int i = 0; i < lenC; ++i) { \
        Y_golden[i] = Val[i] * X[indx[i]];\
    }\
    for (int i = 0; i < iter; ++i) {\
        memset(Y, 0, sizeof(type) * lenC);\
        LineProducts(Val, indx, X, Y);\
    }\
    int cnt = 0;\
    for (int i = 0; i < lenC; ++i) {\
        if (Y[i] != Y_golden[i]) {\
            ++cnt;\
            printf("%f %f\n", Y[i], Y_golden[i]);\
        }\
    }\
    printf("%s_errcnt=  %d in %d\n", funcName, cnt, lenC);\
    free(Y_golden);\
    free(Val);\
    free(Y);\
    free(indx);\
    free(X);                      \
}

Test_LineProduct(double );
Test_LineProduct(float );
#define type double
void testDotProduct(int len){
    int lenC = len;
    type *X = aligned_alloc(ALIEN, sizeof(type) * len);
    int *indx = aligned_alloc(ALIEN, sizeof(int) * len);
    type *Val = aligned_alloc(ALIEN, sizeof(type) * len);
    type Y_golden = 0;
    type *Y = aligned_alloc(ALIEN, sizeof(type) * len);
    for (int i = 0; i < lenC; ++i) {
        indx[i] = lenC - 1 - i;
        X[i] = i + 1;
        Val[i] = i + 1;
    }
    for (int i = 0; i < lenC; ++i) {
        Y_golden+= Val[i] * X[indx[i]];
    }
    dot_product_function func = inner_basic_GetDotProduct(sizeof (type));
    type ret = 0;
    func(len,indx,Val,X,&ret,VECTOR_AVX2);
    //ret  = Dot_d_Products[VECTOR_AVX2](len,indx,Val,X);
    printf("%f %f \n",ret,Y_golden);
}
void testLineProduct(int len){
    int lenC = len;
    type *X = aligned_alloc(ALIEN, sizeof(type) * len);
    int *indx = aligned_alloc(ALIEN, sizeof(int) * len);
    type *Val = aligned_alloc(ALIEN, sizeof(type) * len);
    type *Y_golden = aligned_alloc(ALIEN, sizeof(type) * len);;
    type *Y = aligned_alloc(ALIEN, sizeof(type) * len);
    for (int i = 0; i < lenC; ++i) {
        indx[i] = lenC - 1 - i;
        X[i] = i*0.3 + 1;
        Val[i] = i*0.3 + 1;
    }
    line_product_function func = inner_basic_GetLineProduct(sizeof (type));
    for (int i = 0; i < lenC; ++i) {
        Y_golden[i]= Val[i] * X[indx[i]];
    }
    func (len,Val,indx,X,Y,VECTOR_AVX2);
    int cnt = 0;
    for (int i = 0; i < lenC; ++i) {
        if (Y[i] != Y_golden[i]) {
            ++cnt;
            printf("%f %f\n", Y[i], Y_golden[i]);
        }
    }
    printf("%s_errcnt=  %d in %d\n", "line_product", cnt, lenC);
}
int main(){
    for(int i = 0 ; i < 9 ; ++i){
        Test_double_LineProduct(Line_d_Products_name[i],Line_d_Products[i],4<<(i/3),1 );
        Test_float_LineProduct(Line_s_Products_name[i],Line_s_Products[i],4<<(i/3),1 );
    }


    testDotProduct(128);
    testLineProduct(12822);
}
