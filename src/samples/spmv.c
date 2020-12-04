//
// Created by kouushou on 2020/12/4.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>
#include "mmio_highlevel.h"

int main(int argc, char ** argv)
{
    //freopen("out.txt","w",stdout); //输出重定向，输出数据将保存在out.txt文件中
    char *filename = argv[1];
    printf ("filename = %s\n", filename);

    int iter = atoi(argv[3]);
    printf("#iter is %i \n", iter);
    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);
    printf("#threads is %i \n", nthreads);


    //read matrix
    int m, n, nnzR, isSymmetric;
    mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
    int *RowPtr = (int *)malloc((m+1) * sizeof(int));
    int *ColIdx = (int *)malloc(nnzR * sizeof(int));
    float *Val  = (float *)malloc(nnzR * sizeof(float));
    mmio_data(RowPtr, ColIdx, Val, filename);
    for (int i = 0; i < nnzR; i++)
        Val[i] = 1;
    printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n",m, n, nnzR);
    float *X = (float *)malloc(sizeof(float) * n);
    float *Y = (float *)malloc(sizeof(float) * m);
    memset (X, 0, sizeof(float) * n);
    memset (Y, 0, sizeof(float) * m);
    for (int i = 0; i < n; i++)
        X[i] = 1;
    float *Y_golden = (float *)malloc(sizeof(float) * m);
    memset (Y_golden, 0, sizeof(float) * m);

    for (int i = 0; i < m; i++)
        for(int j = RowPtr[i]; j < RowPtr[i+1]; j++)
            Y_golden[i] += Val[j] * X[ColIdx[j]];

    int S = atoi(argv[4]);//S=4
    int C = atoi(argv[5]);//C=2
    int BloS = floor((float)m / (float)S);//排序的块分成的个数
    printf("BloS=%d\n",BloS);
    int redBloS = m - BloS * S;
    printf("reBloS=%d\n",redBloS);

    int nnzred = RowPtr[m] - RowPtr[BloS * S];//剩余非零元个数
    float *Valred = (float *)malloc(nnzred * sizeof(float));
    int *Colred = (int *)malloc(nnzred * sizeof(int));
    int *Rowred = (int *) malloc(sizeof(int) * nnzred);
    memset (Valred, 0, sizeof(float) * nnzred);
    memset (Colred, 0, sizeof(int) * nnzred);
    memset (Rowred, 0, sizeof(int) * nnzred);
    int b = 0;
    for(int i = BloS * S;i < m;i++)
    {
        for(int j = RowPtr[i];j < RowPtr[i+1];j++)
        {
            Rowred[b] = i;
            Colred[b] = ColIdx[j];
            Valred[b] = Val[j];
            b++;
        }
    }
    /*for(int i = 0;i < nnzred;i++)
    {
        printf("Rowred[%d]=%d\n",i,Rowred[i]);
        printf("Colred[%d]=%d\n",i,Colred[i]);
        printf("Valred[%d]=%1.0f\n",i,Valred[i]);
    }*/

    int p,max = 0;
    for (int i = 0; i < m; i++)
    {
        p = RowPtr[i+1] - RowPtr[i];
        if (max < p)
        {
            max = p;
        }
    }
    //printf("max=%d\n",max);//整体矩阵的每行非零元最大值

    int *RowSort = (int *) malloc(sizeof(int) * (BloS*S));//排序后的行号
    memset (RowSort, 0, sizeof(int) *(BloS*S));
    int *nnzSort = (int *) malloc(sizeof(int) * (BloS*S));//排序后的每行非零元个数
    memset (nnzSort, 0, sizeof(int) *(BloS*S));

    //排序
    for (int k = 0;k < BloS;k++)
    {
        int temp = 0;
        int tempm = 0;
        for(int i = k*S;i < (k+1)*S;i++)
        {
            nnzSort[i] = RowPtr[i+1] - RowPtr[i];
            RowSort[i] = i;
            //printf("nnzSort[%d] = %d\n",i,nnzSort[i]);
            //printf("RowSort[%d] = %d\n",i,RowSort[i]);
        }
        for (int i = 0;i < S - 1;i++)
        {
            for (int j = k*S;j < (k+1)*S - i - 1;j++)
            {
                if(nnzSort[j] < nnzSort[j+1])
                {
                    temp = nnzSort[j];
                    tempm = RowSort[j];
                    nnzSort[j] = nnzSort[j+1];
                    RowSort[j] = RowSort[j+1];
                    nnzSort[j+1] = temp;
                    RowSort[j+1] = tempm;
                }
            }
        }
        /*for(int i = k*S;i < (k+1)*S;i++)
        {
            printf("nnzSort--[%d]=%d\n",i,nnzSort[i]);
            printf("RowSort--[%d]=%d\n",i,RowSort[i]);
        }*/
    }

    float **ValBlo = (float **)malloc(sizeof(float *) * (BloS*S));
    for (int i = 0; i < (BloS*S); ++i)
    {
        ValBlo[i] = (float *)malloc(sizeof(float) * max);
    }
    int **ColBlo = (int **)malloc(sizeof(int *) * (BloS*S));
    for (int i = 0; i < (BloS*S); ++i)
    {
        ColBlo[i] = (int *)malloc(sizeof(int) * max);
    }
    int *Cmax = (int *) malloc(sizeof(int) * (S/C*BloS));
    memset (Cmax, 0, sizeof(float) * (S/C*BloS));

    //存放入二维数组val.col
    for(int k = 0;k < S/C*BloS;k++)
    {
        Cmax[k] = nnzSort[k*C];
        //printf("Cmax[%d]=%d\n",k,Cmax[k]);
        for(int i = 0;i < C;i++)
        {
            for (int j = 0;j < nnzSort[k*C+i]; j++)
            {
                ValBlo[k*C+i][j] = Val[RowPtr[RowSort[k*C+i]] + j];
                ColBlo[k*C+i][j] = ColIdx[RowPtr[RowSort[k*C+i]] + j];

            }
            if(nnzSort[k*C+i] < Cmax[k])
            {
                for(int a = nnzSort[k*C+i];a < Cmax[k];a++)
                {
                    ValBlo[k*C+i][a] = 0;
                    ColBlo[k*C+i][a] = -1;
                }
            }
        }
    }

    //转置存入一维数组
    int ATsum = 0;
    for (int i = 0; i < S/C*BloS; i++)
    {
        int tmp = C * Cmax[i];
        ATsum += tmp;
    }
    //printf("%d\n",ATsum);

    float *ValAT = (float *)malloc(ATsum * sizeof(float));
    int *ColAT = (int *)malloc(ATsum * sizeof(int));
    memset (ValAT, 0, sizeof(float) * ATsum);
    memset (ColAT, 0, sizeof(int) * ATsum);
    int q = 0;

    for(int k = 0;k < S/C*BloS;k++)
    {
        for(int j = 0;j < Cmax[k]; j++)
        {
            for (int i = 0;i < C;i++)
            {
                ValAT[q] = ValBlo[k*C+i][j];
                ColAT[q] = ColBlo[k*C+i][j];
                q++;
            }
        }
    }
    /*
    for(int i = 0;i < ATsum;i++)
    {
        printf("valAT[%d] = %1.0f\n",i,ValAT[i]);
    }
    for(int i = 0;i < ATsum;i++)
    {
        printf("colAT[%d] = %d\n",i,ColAT[i]);
    }
    */

//------------------------------------serial------------------------------------
    struct timeval t1, t2;
    float *Y_first = (float *)malloc(sizeof(float) * (BloS * S));
    memset (Y_first, 0, sizeof(float) * (BloS * S));
    gettimeofday(&t1, NULL);
    int currentiter = 0;
    for (currentiter = 0; currentiter < iter; currentiter++)
    {

        memset (Y_first, 0, sizeof(float) * (BloS * S));
        memset (Y, 0, sizeof(float) * m);
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
            Y[RowSort[i]] = Y_first[i];
        }
        if(redBloS != 0)
        {
            for (int i = 0; i < nnzred; i++)
            {
                float sum = 0;
                sum += Valred[i] * X[Colred[i]];
                Y[Rowred[i]] += sum;
            }
        }
    }
    gettimeofday(&t2, NULL);
    float time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops = 2 * nnzR / time / pow(10,6);
    int errorcount = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i])
            errorcount++;

    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-serial-=-=-=--=-=-=-=-=-=-=-=-\n");
    printf("time = %f\n", time);
    printf("errorcount = %i\n", errorcount);
    printf("GFlops = %f\n", GFlops);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
//------------------------------------------------------------------------

}