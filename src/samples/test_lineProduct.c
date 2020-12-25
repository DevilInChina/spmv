//
// Created by kouushou on 2020/12/6.
//

#include <spmv.h>
#include <stdio.h>
#include <string.h>
#define ALIEN 32

#define type float

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
  ///*
    type *X = aligned_alloc(ALIEN, sizeof(type) * len);
    int *indx = aligned_alloc(ALIEN, sizeof(int) * len);
    type *Val = aligned_alloc(ALIEN, sizeof(type) * len);
    type *Y_golden = aligned_alloc(ALIEN, sizeof(type) * len);;
    type *Y = aligned_alloc(ALIEN, sizeof(type) * len);
/*
    type *X = malloc( sizeof(type) * len);
    int *indx = malloc( sizeof(int) * len);
    type *Val = malloc( sizeof(type) * len);
    type *Y_golden = malloc( sizeof(type) * len);;
    type *Y = malloc( sizeof(type) * len);
*/
    for (int i = 0; i < lenC; ++i) {
        indx[i] = lenC - 1 - i;
        X[i] = i*0.3 + 1;
        Val[i] = i*0.3 + 1;
    }

    memset(Y,0,sizeof(type)*lenC);
    line_product_function func = inner_basic_GetLineProduct(sizeof (type));
    for (int i = 0; i < lenC; ++i) {
        Y_golden[i]= Val[i] * X[indx[i]];
    }
    func (len,Val,indx,X,Y,VECTOR_AVX512);
    int cnt = 0;
    for (int i = 0; i < lenC; ++i) {
        if (Y[i] != Y_golden[i]) {
            ++cnt;
            //printf("%d %f %f\n",i, Y[i], Y_golden[i]);
        }
    }
    printf("%s_errcnt=  %d in %d\n", "line_product", cnt, lenC);

    free(Y);
    free(Y_golden);
    free(Val);
    free(indx);
    free(X);
}
int main(int argc,char **argv) {
    int b = atoi(argv[1]);
    int c = atoi(argv[2]);
    testDotProduct(c);
    for (int i = b; i <= c; ++i) {
        testLineProduct(i);
    }
}
