//
// Created by kouushou on 2021/1/26.
//


#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <string.h>
//#include <arm_neon.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <sys/types.h>
#include <vector>
#include <inttypes.h>
#include "inner_spmv.h"
using namespace std;

#define MAX_LINE_LEN 1000

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef FREQUENCY
#define FREQUENCY 1000
#endif

#ifdef USE_INT_64
typedef int64_t INTTYPE;
#else
typedef int32_t INTTYPE;
#endif

struct COOMATRIX //结构体
{
    int N;
    int NNZ;
    VALUE_TYPE *val;//是stl容器中的数据的数据类型，即迭代器所指对象的类别，在使用stl模板时，需要传入迭代器的参数，这个参数的类别就是容器中数据的类别， 那个参数类型的别名就叫value_type
    int *col_ind;
    int *row_ind;
    int *NNZ_row;

    COOMATRIX(int _NNZ, int _N) {
        int i;
        N = _N;
        NNZ = _NNZ;
        val = (VALUE_TYPE *) malloc(NNZ * sizeof(VALUE_TYPE));
        col_ind = (int *) malloc(NNZ * sizeof(int));
        row_ind = (int *) malloc(NNZ * sizeof(int));
        NNZ_row = (int *) malloc(N * sizeof(int));
        for (i = 0; i < N; i++)
            NNZ_row[i] = 0;
    }

    void PrintMatrix(int NNZ) {
        int i;
        for (i = 0; i < NNZ; i++) {
            printf("(%lf,%d,%d) , ", val[i], row_ind[i], col_ind[i]);
        }
        printf("\n");
    }


    void SortWithVectors() {
        INTTYPE k = 0;
        vector<vector<INTTYPE> > vec_col_ind;//定义vec_col_ind数组，大小为n
        vec_col_ind.resize(N);
        vector<vector<VALUE_TYPE> > vec_val;//定义vec_val数组，大小为n
        vec_val.resize(N);
        for (INTTYPE i = 0; i < NNZ; i++) {
            vec_col_ind[row_ind[i]].push_back(col_ind[i]);//vec_col_ind[row_ind[i]]= col_ind[i]
            vec_val[row_ind[i]].push_back(val[i]);
        }
        for (INTTYPE i = 0; i < N; i++) {
            for (INTTYPE j = 0; j < vec_col_ind[i].size(); j++) {
                //row_ind[k] = i;
                //col_ind[k] = vec_col_ind[i][j];
                //val[k] = vec_val[i][j];
                k++;
            }
        }
    }

};

struct MATRIX_SELL_C_sigma {
    int N;
    int C;
    int NNZ;
    int *NNZ_max;
    VALUE_TYPE *val;
    int *col_ind;
    int *sliceptr;
    int *trpos;

    MATRIX_SELL_C_sigma(int *_NNZ_max, int *_trpos, int _N, int _C) {
        int i, tmp;
        int sum = 0;
        N = _N;
        C = _C;
        NNZ_max = (int *) malloc((N / C + 1) * sizeof(int));
        for (i = 0; i < N / C + 1; i++) {
            NNZ_max[i] = _NNZ_max[i];
        }
        for (i = 0; i < N / C; i++) {
            tmp = C * NNZ_max[i];
            sum += tmp;
        }
        sum = sum + N % C * NNZ_max[N / C];
        val = (VALUE_TYPE *) malloc(sum * sizeof(VALUE_TYPE));
        col_ind = (int *) malloc(sum * sizeof(int));
        sliceptr = (int *) malloc((N / C + 2) * sizeof(int));//存储的分块儿的索引
        trpos = (int *) malloc(N * sizeof(int));
        for (i = 0; i < N; i++)
            trpos[i] = _trpos[i];
    }

    void PrintMatrix() {
        int i;
        int tmp;
        int sum = 0;
        for (i = 0; i < N / C; i++) {
            tmp = C * NNZ_max[i];
            sum += tmp;
        }
        sum = sum + N % C * NNZ_max[N / C];
        printf("Val: \n");
        for (i = 0; i < sum; i++)
            printf("%lf  ", val[i]);
        printf("\n");
        printf("col_ind: \n");
        for (i = 0; i < sum; i++)
            printf("%d  ", col_ind[i]);
        printf("\n");
        printf("sliceptr: \n");
        for (i = 0; i < N / C + 2; i++)
            printf("%d  ", sliceptr[i]);
        printf("\n");
    }
};

void DeleteCOOMATRIX(COOMATRIX *Matrix) {
    free(Matrix->val);
    free(Matrix->col_ind);
    free(Matrix->row_ind);
    free(Matrix->NNZ_row);
}

void DeleteMATRIX_SELL_C_sigma(MATRIX_SELL_C_sigma *Matrix) {
    free(Matrix->val);
    free(Matrix->col_ind);
    free(Matrix->sliceptr);
    free(Matrix->NNZ_max);
    free(Matrix->trpos);
}


int SearchMax_int(int *arr, int N) //最大值
{
    int i;
    int max_arr = arr[0];
    for (i = 1; i < N; i++) {
        if (max_arr < arr[i])
            max_arr = arr[i];
    }
    return max_arr;
}

VALUE_TYPE SearchMax_VALUE_TYPE(VALUE_TYPE *arr, int N) //？？？
{
    int i;
    VALUE_TYPE max_arr = arr[0];
    for (i = 1; i < N; i++) {
        if (max_arr < arr[i])
            max_arr = arr[i];
    }
    return max_arr;
}

void TransposeRow(int *NNZ_row, int N, int *trpos)       //按非零元数量从大到小排列NNZ_row和trops，NNZ_row：每一行的非零元个数
{
    int *NNZ_row_copy = (int *) malloc(N * sizeof(int));
    int i, j, tmp1, tmp2;
    for (i = 0; i < N; i++) {
        NNZ_row_copy[i] = NNZ_row[i];
    }
    for (i = 0; i < N - 1; i++)
        for (j = i + 1; j < N; j++) {
            if (NNZ_row_copy[i] < NNZ_row_copy[j]) {
                tmp1 = NNZ_row_copy[i];
                NNZ_row_copy[i] = NNZ_row_copy[j];
                NNZ_row_copy[j] = tmp1;

                tmp2 = trpos[i];
                trpos[i] = trpos[j];
                trpos[j] = tmp2;
            }
        }
    free(NNZ_row_copy);
}

MATRIX_SELL_C_sigma *ConverterInSELL_C_sigma(COOMATRIX *Matrix, int C, int sigma)//sigma为排序范围，8，512
{
    int i, j, max_r, k, p, q, tmp1, tmp2, tmp2_C, N_tr;
    int sigmaDivideC = sigma / C;//一个排序里分64块
    int m = 0;
    int m_C = 0;
    int t = 0;
    int *NNZ_max_C;
    int *NNZ_max_sigma;
    VALUE_TYPE **val_e;
    VALUE_TYPE **val_e_trpos = NULL;
    int **col_ind_e_trpos = NULL;
    int **col_ind_e;
    int *trpos = (int *) malloc(Matrix->N * sizeof(int));
    for (i = 0; i < Matrix->N; i++)//索引
        trpos[i] = i;
    N_tr = Matrix->N / sigma;//排序的块数
    for (i = 0; i < N_tr; i++)
        TransposeRow(Matrix->NNZ_row + i * sigma, sigma, trpos + i * sigma);//分出排序范围sigma，vsluse排序
    if (Matrix->N % sigma != 0)//排序有余
        TransposeRow(Matrix->NNZ_row + N_tr * sigma, Matrix->N % sigma, trpos + N_tr * sigma);
    NNZ_max_C = (int *) malloc((Matrix->N / C + 1) * sizeof(int));
    NNZ_max_sigma = (int *) malloc((Matrix->N / sigma + 1) * sizeof(int));
    int *NNZ_row_trpos = (int *) malloc(Matrix->N * sizeof(int));//原索引
    memset(NNZ_max_C, 0, (Matrix->N / C + 1) * sizeof(int));
    memset(NNZ_max_sigma, 0, (Matrix->N / sigma + 1) * sizeof(int));
    memset(NNZ_row_trpos, 0, Matrix->N * sizeof(int));
    for (i = 0; i < Matrix->N; i++)
        NNZ_row_trpos[i] = Matrix->NNZ_row[trpos[i]];

    for (i = 0; i < Matrix->N / C; i++)
        NNZ_max_C[i] = SearchMax_int(NNZ_row_trpos + i * C, C);//对c索引排序
    if (Matrix->N % C != 0)
        NNZ_max_C[Matrix->N / C] = SearchMax_int(NNZ_row_trpos + (Matrix->N / C) * C, Matrix->N % C);//处理不整除部分

    for (i = 0; i < Matrix->N / sigma; i++)
        NNZ_max_sigma[i] = SearchMax_int(NNZ_row_trpos + i * sigma, sigma);

    if (Matrix->N % sigma != 0)//有余
    {
        NNZ_max_sigma[Matrix->N / sigma] = SearchMax_int(NNZ_row_trpos + (Matrix->N / sigma) * sigma,
                                                         Matrix->N % sigma);//对于某些矩阵出错
        max_r = SearchMax_int(NNZ_max_sigma, Matrix->N / sigma + 1);
    } else
        max_r = SearchMax_int(NNZ_max_sigma, Matrix->N / sigma);//max_r = 用最长行对所有的sigma排序从大到小
    cout << "max_r = " << max_r << endl;
    val_e = (VALUE_TYPE **) malloc(sigma * sizeof(VALUE_TYPE *));
    for (i = 0; i < sigma; i++)
        val_e[i] = (VALUE_TYPE *) malloc(max_r * sizeof(VALUE_TYPE));
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    col_ind_e = (int **) malloc(sigma * sizeof(int *));
    for (i = 0; i < sigma; i++)
        col_ind_e[i] = (int *) malloc(max_r * sizeof(int));
    MATRIX_SELL_C_sigma *Matrix_conv = new MATRIX_SELL_C_sigma(NNZ_max_C, trpos, Matrix->N, C);
    k = 0;
    Matrix_conv->sliceptr[0] = 0;
    //开销大
    while (m_C != Matrix->N / C + 1)//在每一补零块
    {
        if (m_C != Matrix->N / C)//有补零块数，整除
            tmp2_C = C;//tmp2_C=8
        else
            tmp2_C = Matrix->N % C;//有<8的块，tmp2_C =7，6，。。。。
        Matrix_conv->sliceptr[m_C + 1] = Matrix_conv->sliceptr[m_C] + NNZ_max_C[m_C] * tmp2_C;//
        m_C++;//0-65
    }

    while (m != Matrix->N / sigma + 1)//排序的
    {
        p = 0;
        if (m != Matrix->N / sigma)
            tmp1 = (m + 1) * sigma;//用tmp1给sigma建一维数组
        else
            tmp1 = m * sigma + Matrix->N % sigma;//在整的前面加上余下的sigma

        for (i = 0; i < sigma; i++)//在每一个sigma中
        {
            for (j = 0; j < max_r; j++) {
                val_e[i][j] = 0;
                col_ind_e[i][j] = -1;
            }
        }

        for (i = m * sigma; i < tmp1; i++)//每一个sigma的开始到结束位置
        {
            while ((t < Matrix->NNZ) && (Matrix->row_ind[t] == i))//按行由小到大，列由小到大排列的顺序存储,每块有C行
            {
                val_e[i - m * sigma][p] = Matrix->val[t];
                col_ind_e[i - m * sigma][p] = Matrix->col_ind[t];
                p++;
                t++;
            }
            p = 0;
        }
        val_e_trpos = (VALUE_TYPE **) malloc(sigma * sizeof(VALUE_TYPE *));
        for (i = 0; i < sigma; i++)
            val_e_trpos[i] = (VALUE_TYPE *) malloc(max_r * sizeof(VALUE_TYPE));

        col_ind_e_trpos = (int **) malloc(sigma * sizeof(int *));
        for (i = 0; i < sigma; i++)
            col_ind_e_trpos[i] = (int *) malloc(max_r * sizeof(int));

        for (i = 0; i < sigma; i++)
            for (j = 0; j < max_r; j++) {
                val_e_trpos[i][j] = 0;
                col_ind_e_trpos[i][j] = -1;
            }
        for (i = 0; i < sigma; i++)
            for (j = 0; j < max_r; j++)//由max_r改为的NNZ_row_trpos[i + sigma * m]
                if (i + sigma * m < Matrix->N) {
                    val_e_trpos[i][j] = val_e[trpos[i + sigma * m] - sigma *
                                                                     m][j];//当sigma>4时trpos[i + C * m] - C * m会出现负值!!!!!!!!!!!!!!!!!!!!!!!!
                    col_ind_e_trpos[i][j] = col_ind_e[trpos[i + sigma * m] - sigma * m][j];
                }

//*************************改为列存储*******************************
        if (sigma > C) {
            if (m != (Matrix->N / sigma))//整除部分
                for (q = 0; q < sigmaDivideC; q++)
                    for (i = 0; i < NNZ_max_C[sigmaDivideC * m + q]; i++)
                        for (j = q * C; j < (q + 1) * C; j++) {
                            Matrix_conv->val[k] = val_e_trpos[j][i];
                            Matrix_conv->col_ind[k] = col_ind_e_trpos[j][i];//val_e_trpos在sigma块中的第二个C块时没换行
                            k++;
                        }
            else//处理不整除部分
            {
                if (Matrix->N % sigma < C)//当余数小于C
                    for (i = 0; i < NNZ_max_C[sigmaDivideC * m]; i++)
                        for (j = 0; j < Matrix->N % C; j++) {
                            Matrix_conv->val[k] = val_e_trpos[j][i];
                            Matrix_conv->col_ind[k] = col_ind_e_trpos[j][i];
                            k++;
                        }
                else//余数大于或等于C
                {
                    for (q = 0; q < (Matrix->N % sigma) / C; q++)
                        for (i = 0; i < NNZ_max_C[sigmaDivideC * m + q]; i++)
                            for (j = q * C; j < (q + 1) * C; j++) {
                                Matrix_conv->val[k] = val_e_trpos[j][i];
                                Matrix_conv->col_ind[k] = col_ind_e_trpos[j][i];
                                k++;
                            }
                    for (i = 0; i < NNZ_max_C[sigmaDivideC * m + (Matrix->N % sigma) / C]; i++)
                        for (j = q * C; j < q * C + Matrix->N % C; j++) {
                            Matrix_conv->val[k] = val_e_trpos[j][i];
                            Matrix_conv->col_ind[k] = col_ind_e_trpos[j][i];
                            k++;
                        }
                }
            }
        } else//sigma = C
        {
            if (m != Matrix->N / sigma)
                tmp2 = C;
            else
                tmp2 = Matrix->N % C;
            for (i = 0; i < NNZ_max_C[m]; i++)
                for (j = 0; j < tmp2; j++) {
                    Matrix_conv->val[k] = val_e_trpos[j][i];
                    Matrix_conv->col_ind[k] = col_ind_e_trpos[j][i];
                    k++;
                }
        }
//************************改为列存储*********************************
        m++;
        for (i = 0; i < sigma; i++) {
            free(val_e_trpos[i]);
            free(col_ind_e_trpos[i]);
        }
        free(val_e_trpos);
        free(col_ind_e_trpos);
    }

    free(NNZ_max_C);
    free(NNZ_max_sigma);
    for (i = 0; i < sigma; i++) {
        free(val_e[i]);
        free(col_ind_e[i]);
    }
    free(val_e);
    free(col_ind_e);
    free(trpos);
    free(NNZ_row_trpos);
    return Matrix_conv;
}

/*********************************************performance_test*******************************************************/


struct anonymouslib_timer {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }

    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};

/**********************************************performance_test*****************************************************************************/
double
Matrix_VectorMultiplicationInSELL_C_sigma(MATRIX_SELL_C_sigma *Matrix, VALUE_TYPE *Vector, int N, VALUE_TYPE *result) {
    int i, j;
    int k = 0;
    int m = 0;
    int N_Block;//n块
    omp_set_num_threads(8);
    VALUE_TYPE vec_tmp_0, vec_tmp_1, vec_tmp_2, vec_tmp_3;//？？
    vec_tmp_0 = vec_tmp_1 = vec_tmp_2 = vec_tmp_3 = 1.0;
    anonymouslib_timer run_timer;
    for (i = 0; i < N; i++) {
        result[i] = 0;
    }
//********************************************spmv********************************************
    run_timer.start();
#pragma omp parallel for //private()
    for (i = 0; i < N / Matrix->C; i++)//进入补零的每块C（对512行进行排序，每8个进行填充）
    {
        int offsets = Matrix->sliceptr[i];//存储的分块儿的索引
        VALUE_TYPE vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g, vec_h;
        vec_a = vec_b = vec_c = vec_d = vec_e = vec_f = vec_g = vec_h = 0.0;
        for (int j = 0; j < Matrix->NNZ_max[i]; j++)//0-最长行：NNZ_max：每一块C的最长行的数；vec：每行进行计算
        {
            vec_a += Matrix->val[offsets + j * Matrix->C + 0] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 0]];
            //Matrix -> C = 8
            vec_b += Matrix->val[offsets + j * Matrix->C + 1] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 1]];
            vec_c += Matrix->val[offsets + j * Matrix->C + 2] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 2]];
            vec_d += Matrix->val[offsets + j * Matrix->C + 3] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 3]];
            vec_e += Matrix->val[offsets + j * Matrix->C + 4] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 4]];
            vec_f += Matrix->val[offsets + j * Matrix->C + 5] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 5]];
            vec_g += Matrix->val[offsets + j * Matrix->C + 6] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 6]];
            vec_h += Matrix->val[offsets + j * Matrix->C + 7] * Vector[Matrix->col_ind[offsets + j * Matrix->C + 7]];
        }
        result[Matrix->trpos[i * Matrix->C + 0]] += vec_a;
        result[Matrix->trpos[i * Matrix->C + 1]] += vec_b;
        result[Matrix->trpos[i * Matrix->C + 2]] += vec_c;
        result[Matrix->trpos[i * Matrix->C + 3]] += vec_d;
        result[Matrix->trpos[i * Matrix->C + 4]] += vec_e;
        result[Matrix->trpos[i * Matrix->C + 5]] += vec_f;
        result[Matrix->trpos[i * Matrix->C + 6]] += vec_g;
        result[Matrix->trpos[i * Matrix->C + 7]] += vec_h;
    }
    if (N % Matrix->C != 0)
        for (j = 0; j < Matrix->NNZ_max[N / Matrix->C]; ++j)//最后一个块的最长行长度
            for (k = 0; k < N % Matrix->C; k++)//按最后一个块的高度从上往下循环
                result[Matrix->trpos[(N / Matrix->C) * Matrix->C + k]] +=
                        Matrix->val[Matrix->sliceptr[N / Matrix->C] + j * (N % Matrix->C) + k] *
                        Vector[Matrix->col_ind[Matrix->sliceptr[N / Matrix->C] + j * (N % Matrix->C) + k]];
    double run_time = run_timer.stop();
    //cout<<"run_time = "<<run_time<<endl;
    return run_time;
}
void sell_C_Sigma_get_handle_Selected(spmv_Handle_t handle,
                                      BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE *RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const void *Matrix_Val
) {

}