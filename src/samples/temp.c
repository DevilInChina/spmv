#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>
#include "mmio_highlevel.h"
#include <spmv.h>

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int key_input,
                                        const int size) {
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start) {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

int main(int argc, char **argv) {
    //freopen("out.txt","w",stdout); //输出重定向，输出数据将保存在out.txt文件中
    char *filename = argv[1];
    printf("filename = %s\n", filename);

    //read matrix
    int m, n, nnzR, isSymmetric;

    mmio_info(&m, &n, &nnzR, &isSymmetric, filename);
    int *RowPtr = (int *) malloc((m + 1) * sizeof(int));
    int *ColIdx = (int *) malloc(nnzR * sizeof(int));
    float *Val = (float *) malloc(nnzR * sizeof(float));
    mmio_data(RowPtr, ColIdx, Val, filename);
    for (int i = 0; i < nnzR; i++)
        Val[i] = 1;
    printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n", m, n, nnzR);

    //create X, Y,Y_golden
    float *X = (float *) malloc(sizeof(float) * (n + 1));
    float *Y = (float *) malloc(sizeof(float) * (m + 1));
    float *Y_golden = (float *) malloc(sizeof(float) * (m + 1));

    memset(X, 0, sizeof(float) * (n + 1));
    memset(Y, 0, sizeof(float) * (m + 1));
    memset(Y_golden, 0, sizeof(float) * (m + 1));

    for (int i = 0; i < n; i++)
        X[i] = 1;

    for (int i = 0; i < m; i++)
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
            Y_golden[i] += Val[j] * X[ColIdx[j]];

    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);
    printf("#threads is %i \n", nthreads);

    int iter = atoi(argv[3]);
    //printf("#iter is %i \n", iter);

    // find balanced points
    int *csrSplitter = (int *) malloc((nthreads + 1) * sizeof(int));
    int stridennz = ceil((float) nnzR / (float) nthreads);

#pragma omp parallel for
    for (int tid = 0; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary_yid = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary_yid = boundary_yid > nnzR ? nnzR : boundary_yid;
        // binary search
        csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary_yid, m + 1) - 1;
        //printf("csrSplitter[%d] is %d\n", tid, csrSplitter[tid]);

    }

    int *Apinter = (int *) malloc(nthreads * sizeof(int));
    memset(Apinter, 0, nthreads * sizeof(int));
    //每个线程执行行数
    for (int tid = 0; tid < nthreads; tid++) {
        Apinter[tid] = csrSplitter[tid + 1] - csrSplitter[tid];
        //printf("A[%d] is %d\n", tid, Apinter[tid]);
    }

    int *Bpinter = (int *) malloc(nthreads * sizeof(int));
    memset(Bpinter, 0, nthreads * sizeof(int));
    //每个线程执行非零元数
    for (int tid = 0; tid < nthreads; tid++) {
        int num = 0;
        for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
            num += RowPtr[u + 1] - RowPtr[u];
        }
        Bpinter[tid] = num;
        //printf("B [%d]is %d\n",tid, Bpinter[tid]);
    }

    int *Yid = (int *) malloc(sizeof(int) * nthreads);
    memset(Yid, 0, sizeof(int) * nthreads);
    //每个线程
    int flag;
    for (int tid = 0; tid < nthreads; tid++) {
        //printf("tid = %i, csrSplitter: %i -> %i\n", tid, csrSplitter[tid], csrSplitter[tid+1]);
        if (csrSplitter[tid + 1] - csrSplitter[tid] == 0) {
            Yid[tid] = csrSplitter[tid];
            flag = 1;
        }
        if (csrSplitter[tid + 1] - csrSplitter[tid] != 0) {
            Yid[tid] = -1;
        }
        if (csrSplitter[tid + 1] - csrSplitter[tid] != 0 && flag == 1) {
            Yid[tid] = csrSplitter[tid];
            flag = 0;
        }
        //printf("Yid[%d] is %d\n", tid, Yid[tid]);
    }

    //行平均用在多行上
    //int sto = nthreads > nnzR ? nthreads : nnzR;
    int *Start1 = (int *) malloc(sizeof(int) * nthreads);
    memset(Start1, 0, sizeof(int) * nthreads);
    int *End1 = (int *) malloc(sizeof(int) * nthreads);
    memset(End1, 0, sizeof(int) * nthreads);
    int *label = (int *) malloc(sizeof(int) * nthreads);
    memset(label, 0, sizeof(int) * nthreads);

    int start1, search1 = 0;
    for (int tid = 0; tid < nthreads; tid++) {
        if (Apinter[tid] == 0) {
            if (search1 == 0) {
                start1 = tid;
                search1 = 1;
            }
        }
        if (search1 == 1 && Apinter[tid] != 0) {
            int nntz = floor((float) Apinter[tid] / (float) (tid - start1 + 1));
            if (nntz != 0) {
                for (int i = start1; i <= tid; i++) {
                    label[i] = i;
                }
            } else if ((tid - start1 + 1) >= Apinter[tid] && Apinter[tid] != 0) {
                for (int i = start1; i <= tid; i++) {
                    label[i] = i;
                }
            }
            int mntz = Apinter[tid] - (nntz * (tid - start1));
            //start and end
            int n = start1;
            Start1[n] = csrSplitter[tid];
            End1[n] = Start1[n] + nntz;
            //printf("start1a[%d] = %d, end1a[%d] = %d\n",n,Start1[n],n, End1[n]);
            for (int p = start1 + 1; p <= tid; p++) {
                if (p == tid) {
                    Start1[p] = End1[p - 1];
                    End1[p] = Start1[p] + mntz;
                } else {
                    Start1[p] = End1[p - 1];
                    End1[p] = Start1[p] + nntz;
                }
                //printf("start1b[%d] = %d, end1b[%d] = %d\n",n,Start1[n],n, End1[n]);
            }
            search1 = 0;
        }
    }
    dot_product_function dotProductFunction = inner_basic_GetDotProduct(sizeof(VALUE_TYPE));
    //非零元平均用在一行
    VALUE_TYPE *Ypartialsum = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * nthreads);
    memset(Ypartialsum, 0, sizeof(VALUE_TYPE) * nthreads);
    VALUE_TYPE *Ysum = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * nthreads);
    memset(Ysum, 0, sizeof(VALUE_TYPE) * nthreads);

    int *Start2 = (int *) malloc(sizeof(int) * nthreads);
    memset(Start2, 0, sizeof(int) * nthreads);
    int *End2 = (int *) malloc(sizeof(int) * nthreads);
    memset(End2, 0, sizeof(int) * nthreads);
    int start2, search2 = 0;
    for (int tid = 0; tid < nthreads; tid++) {
        if (Bpinter[tid] == 0) {
            if (search2 == 0) {
                start2 = tid;
                search2 = 1;
            }
        }
        if (search2 == 1 && Bpinter[tid] != 0) {
            int nntz2 = ceil((double ) Bpinter[tid] / (double ) (tid - start2 + 1));
            int mntz2 = Bpinter[tid] - (nntz2 * (tid - start2));
            //start and end
            int n = start2;
            for (int i = start2; i >= 0; i--) {
                Start2[n] += Bpinter[i];
                End2[n] = Start2[n] + nntz2;
                //printf("starta[%d] = %d, enda[%d] = %d\n",n,Start2[n],n, End2[n]);
            }
            //printf("starta[%d] = %d, enda[%d] = %d\n",n,Start2[n],n, End2[n]);
            for (n = start2 + 1; n < tid; n++) {
                Start2[n] = End2[n - 1];
                End2[n] = Start2[n] + nntz2;
                //printf("startb[%d] = %d, endb[%d] = %d\n",n,Start2[n],n, End2[n]);
            }
            //printf("startb[%d] = %d, endb[%d] = %d\n",n,Start2[n],n, End2[n]);
            if (n == tid) {
                Start2[n] = End2[n - 1];
                End2[n] = Start2[n] + mntz2;
                //printf("startc[%d] = %d, endc[%d] = %d\n",n,Start2[n],n, End2[n]);
            }
            //printf("startc[%d] = %d, endc[%d] = %d\n",n,Start2[n],n, End2[n]);
            search2 = 0;
        }
    }

    struct timeval t1, t2;
    int currentiter = 0;
    gettimeofday(&t1, NULL);
    const int way = VECTOR_NONE;
    for (currentiter = 0; currentiter < iter; currentiter++) {

        for (int tid = 0; tid < nthreads; tid++) {
            if (Yid[tid] != -1)
                Y[Yid[tid]] = 0;
            Ysum[tid] = 0;
        }
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            if (Yid[tid] == -1) {
                for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                    dotProductFunction(
                            RowPtr[u + 1]- RowPtr[u],
                            ColIdx+ RowPtr[u],
                            Val+RowPtr[u],
                            X,Y+u,way
                    );
                }
            }
            else if (label[tid] != 0) {
                for (int u = Start1[tid]; u < End1[tid]; u++) {

                    dotProductFunction(
                            RowPtr[u + 1]- RowPtr[u],
                            ColIdx+ RowPtr[u],
                            Val+RowPtr[u],
                            X,Y+u,way
                    );
                }
            }
            else if (Yid[tid] != -1 && label[tid] == 0) {
                //Ysum[tid] = 0;
                dotProductFunction(
                        End2[tid] - Start2[tid],
                        ColIdx + Start2[tid], Val + Start2[tid], X,
                        Ypartialsum+tid,way
                );
                //((double *)Ysum)[tid] += ((double *)Ypartialsum)[tid];
                Ysum[tid]+=Ypartialsum[tid];
            }
        }
        for(int tid = 0 ; tid < nthreads ; ++tid){
            if(Yid[tid]!=-1) {
                Y[Yid[tid]]+=Ysum[tid];
            }
        }
    }
    gettimeofday(&t2, NULL);
    float time_overall_parallel_omp_balanced_Yid =
            ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops_parallel_omp_balanced_Yid = 2 * nnzR / time_overall_parallel_omp_balanced_Yid / pow(10, 6);
    int errorcount_parallel_omp_balanced_Yid = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i]) {
            printf("Y[%d] = %f\n", i, Y[i]);
            printf("Y_golden[%d] = %f\n", i, Y_golden[i]);
            errorcount_parallel_omp_balanced_Yid++;
        }

    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-parallel_omp-=-=-=--=-=-=-=-=-=-=-=-\n");
    //printf("time_overall_parallel = %f\n", time_overall_parallel);
    printf("errorcount_parallel_omp_balanced_Yid = %i\n", errorcount_parallel_omp_balanced_Yid);
    printf("GFlops_parallel_omp_balanced_Yid = %f\n", GFlops_parallel_omp_balanced_Yid);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
    //free(Y);//加一
}