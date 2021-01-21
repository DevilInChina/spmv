#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>
#include "mmio_highlevel.h"

// sum up 8 single-precision numbers
float hsum_avx(__m256 in256) {
    float sum;

    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sum, _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));

    return sum;
}

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
    //int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));
    int stridennz = ceil((double) nnzR / (double) nthreads);

#pragma omp parallel for
    for (int tid = 0; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;

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
    int sto = nthreads > nnzR ? nthreads : nnzR;
    int *Start1 = (int *) malloc(sizeof(int) * sto);
    memset(Start1, 0, sizeof(int) * sto);
    int *End1 = (int *) malloc(sizeof(int) * sto);
    memset(End1, 0, sizeof(int) * sto);
    int start1, search1 = 0;
    for (int tid = 0; tid < nthreads; tid++) {
        if (Apinter[tid] == 0) {
            if (search1 == 0) {
                start1 = tid;
                search1 = 1;
            }
        }
        if (search1 == 1 && Apinter[tid] != 0) {
            int nntz = ceil((double) Apinter[tid] / (double) (tid - start1 + 1));
            int mntz = Apinter[tid] - (nntz * (tid - start1));
            //start and end
            int n = start1;
            Start1[n] = csrSplitter[tid];
            End1[n] = Start1[n] + nntz;
            //printf("start1a[%d] = %d, end1a[%d] = %d\n",n,Start1[n],n, End1[n]);
            for (n = start1 + 1; n < tid; n++) {
                Start1[n] = End1[n - 1];
                End1[n] = Start1[n] + nntz;
                //printf("start1b[%d] = %d, end1b[%d] = %d\n",n,Start1[n],n, End1[n]);
            }
            if (n == tid) {
                Start1[n] = End1[n - 1];
                End1[n] = Start1[n] + mntz;
                //printf("start1c[%d] = %d, end1c[%d] = %d\n",n,Start1[n],n, End1[n]);
            }
            //printf("start1c[%d] = %d, end1c[%d] = %d\n",n,Start1[n],n, End1[n]);
            for (int j = start1; j <= tid - 1; j++) {
                Apinter[j] = nntz;
            }
            Apinter[tid] = mntz;
            search1 = 0;
        }
    }
    //非零元平均用在一行
    float *Ypartialsum = (float *) malloc(sizeof(float) * nthreads);
    memset(Ypartialsum, 0, sizeof(float) * nthreads);
    float *Ysum = (float *) malloc(sizeof(float) * nthreads);
    memset(Ysum, 0, sizeof(float) * nthreads);
    int *Start2 = (int *) malloc(sizeof(int) * sto);
    memset(Start2, 0, sizeof(int) * sto);
    int *End2 = (int *) malloc(sizeof(int) * sto);
    memset(End2, 0, sizeof(int) * sto);
    int start2, search2 = 0;
    for (int tid = 0; tid < nthreads; tid++) {
        if (Bpinter[tid] == 0) {
            if (search2 == 0) {
                start2 = tid;
                search2 = 1;
            }
        }
        if (search2 == 1 && Bpinter[tid] != 0) {
            int nntz2 = ceil((double) Bpinter[tid] / (double) (tid - start2 + 1));
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

//------------------------------------serial--------------------------------
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    int currentiter = 0;
    for (currentiter = 0; currentiter < iter; currentiter++) {
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++) {
                sum += Val[j] * X[ColIdx[j]];
            }
            Y[i] = sum;
        }
    }
    gettimeofday(&t2, NULL);
    float time_overall_serial = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
    int errorcount_serial = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i])
            errorcount_serial++;

    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=serial-=-=-=-=-=-=-=-=-=-=-=-=-=-\n");
    //printf("time_overall_serial = %f\n", time_overall_serial);
    printf("errorcount_serial = %i\n", errorcount_serial);
    printf("GFlops_serial = %f\n", GFlops_serial);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
    //free(Y);//加一
//------------------------------------------------------------------------

//-----------------------------------parallel_omp-------------------------------------
    gettimeofday(&t1, NULL);
    for (currentiter = 0; currentiter < iter; currentiter++) {
#pragma omp parallel for
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++) {
                sum += Val[j] * X[ColIdx[j]];
            }
            Y[i] = sum;
        }
    }
    gettimeofday(&t2, NULL);
    float time_overall_parallel =
            ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops_parallel = 2 * nnzR / time_overall_parallel / pow(10, 6);
    int errorcount_parallel = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i])
            errorcount_parallel++;

    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-parallel_omp-=-=-=--=-=-=-=-=-=-=-=-\n");
    //printf("time_overall_parallel = %f\n", time_overall_parallel);
    printf("errorcount_parallel = %i\n", errorcount_parallel);
    printf("GFlops_parallel = %f\n", GFlops_parallel);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
    //free(Y);//加一
//------------------------------------------------------------------------

//-----------------------------------parallel_omp_balanced-------------------------------------
    gettimeofday(&t1, NULL);
    for (currentiter = 0; currentiter < iter; currentiter++) {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                float sum = 0;
                for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) {
                    sum += Val[j] * X[ColIdx[j]];
                }
                Y[u] = sum;
            }
        }
    }
    gettimeofday(&t2, NULL);
    float time_overall_parallel_balanced =
            ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops_parallel_balanced = 2 * nnzR / time_overall_parallel_balanced / pow(10, 6);
    int errorcount_parallel_balanced = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i])
            errorcount_parallel_balanced++;

    //printf("omp_balanced length = %d\n", length);
    //printf("-=-=-=-=-=-=-=-=-=-=parallel_omp_balanced-=-=-=-=-=-=-=-=-=-=\n");
    //printf("time_overall_parallel_balanced = %f\n", time_overall_parallel_balanced);
    printf("errorcount_parallel_balanced = %i\n", errorcount_parallel_balanced);
    printf("GFlops_parallel_balanced = %f\n", GFlops_parallel_balanced);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
    //free(Y);//加一
//------------------------------------------------------------------------

//-----------------------------------parallel_omp_balanced_Yid-------------------------------------
    gettimeofday(&t1, NULL);
    for (currentiter = 0; currentiter < iter; currentiter++) {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
            Y[Yid[tid]] = 0;
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            if (Yid[tid] == -1) {
                for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                    float sum = 0;
                    for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) {
                        sum += Val[j] * X[ColIdx[j]];
                    }
                    Y[u] = sum;
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] > 1) {
                for (int u = Start1[tid]; u < End1[tid]; u++) {
                    float sum = 0;
                    for (int j = RowPtr[u]; j < RowPtr[u + 1]; j++) {
                        sum += Val[j] * X[ColIdx[j]];
                    }
                    Y[u] = sum;
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] <= 1) {
                Ysum[tid] = 0;
                Ypartialsum[tid] = 0;
                for (int j = Start2[tid]; j < End2[tid]; j++) {
                    Ypartialsum[tid] += Val[j] * X[ColIdx[j]];
                }
                Ysum[tid] += Ypartialsum[tid];
                Y[Yid[tid]] += Ysum[tid];
            }
        }
    }
    gettimeofday(&t2, NULL);
    float time_overall_parallel_omp_balanced_Yid =
            ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops_parallel_omp_balanced_Yid = 2 * nnzR / time_overall_parallel_omp_balanced_Yid / pow(10, 6);
    int errorcount_parallel_omp_balanced_Yid = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i])
            errorcount_parallel_omp_balanced_Yid++;

    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-parallel_omp-=-=-=--=-=-=-=-=-=-=-=-\n");
    //printf("time_overall_parallel = %f\n", time_overall_parallel);
    printf("errorcount_parallel_omp_balanced_Yid = %i\n", errorcount_parallel_omp_balanced_Yid);
    printf("GFlops_parallel_omp_balanced_Yid = %f\n", GFlops_parallel_omp_balanced_Yid);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
    //free(Y);//加一
//-----------------------------------------------------------------------





//------------------------------------parallel_omp_balanced_avx2_Yid------------------------------------
    gettimeofday(&t1, NULL);
    for (currentiter = 0; currentiter < iter; currentiter++) {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
            Y[Yid[tid]] = 0;
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            if (Yid[tid] == -1) {
                for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                    float sum = 0;
                    __m256 res = _mm256_setzero_ps();
                    int dif = RowPtr[u + 1] - RowPtr[u];
                    int nloop = dif / 8;
                    int remainder = dif % 8;
                    for (int li = 0; li < nloop; li++) {
                        int j = RowPtr[u] + li * 8;
                        __m256 vecv = _mm256_loadu_ps(&Val[j]);
                        __m256i veci = _mm256_loadu_si256((__m256i *) (&ColIdx[j]));
                        __m256 vecx = _mm256_i32gather_ps(X, veci, 4);
                        res = _mm256_fmadd_ps(vecv, vecx, res);
                    }
                    //Y[u] += _mm256_reduce_add_ps(res);
                    sum += hsum_avx(res);

                    for (int j = RowPtr[u] + nloop * 8; j < RowPtr[u + 1]; j++) {
                        sum += Val[j] * X[ColIdx[j]];
                    }
                    Y[u] = sum;
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] > 1) {
                for (int u = Start1[tid]; u < End1[tid]; u++) {
                    float sum = 0;
                    __m256 res = _mm256_setzero_ps();
                    int dif = RowPtr[u + 1] - RowPtr[u];
                    int nloop = dif / 8;
                    int remainder = dif % 8;
                    for (int li = 0; li < nloop; li++) {
                        int j = RowPtr[u] + li * 8;
                        __m256 vecv = _mm256_loadu_ps(&Val[j]);
                        __m256i veci = _mm256_loadu_si256((__m256i *) (&ColIdx[j]));
                        __m256 vecx = _mm256_i32gather_ps(X, veci, 4);
                        res = _mm256_fmadd_ps(vecv, vecx, res);
                    }
                    //Y[u] += _mm256_reduce_add_ps(res);
                    sum += hsum_avx(res);

                    for (int j = RowPtr[u] + nloop * 8; j < RowPtr[u + 1]; j++) {
                        sum += Val[j] * X[ColIdx[j]];
                    }
                    Y[u] = sum;
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] <= 1) {
                Ysum[tid] = 0;
                Ypartialsum[tid] = 0;
                __m256 res = _mm256_setzero_ps();
                int dif = End2[tid] - Start2[tid];
                int nloop = dif / 8;
                int remainder = dif % 8;
                for (int j = 0; j < nloop; j++) {
                    __m256 vecv = _mm256_loadu_ps(&Val[j]);
                    __m256i veci = _mm256_loadu_si256((__m256i *) (&ColIdx[j]));
                    __m256 vecx = _mm256_i32gather_ps(X, veci, 4);
                    res = _mm256_fmadd_ps(vecv, vecx, res);
                }
                Ypartialsum[tid] += hsum_avx(res);
                for (int j = Start2[tid] + nloop * 8; j < End2[tid]; j++) {
                    Ypartialsum[tid] += Val[j] * X[ColIdx[j]];
                }
                Ysum[tid] += Ypartialsum[tid];
                Y[Yid[tid]] += Ysum[tid];
            }
        }
    }
    gettimeofday(&t2, NULL);
    float time_overall_parallel_avx2_Yid =
            ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / atoi(argv[3]);
    float GFlops_parallel_avx2_Yid = 2 * nnzR / time_overall_parallel_avx2_Yid / pow(10, 6);
    int errorcount_parallel_avx2_Yid = 0;
    for (int i = 0; i < m; i++)
        if (Y[i] != Y_golden[i])
            errorcount_parallel_avx2_Yid++;

    //printf("-=-=-=-=-=-=-=-=parallel_omp_balanced_avx2-=-=-=-=-=-=-=-=-=-\n");
    //printf("time_overall_parallel_avx2 = %f\n", time_overall_parallel_avx2);
    printf("errorcount_parallel_avx2_Yid = %i\n", errorcount_parallel_avx2_Yid);
    printf("GFlops_parallel_avx2_Yid = %f\n", GFlops_parallel_avx2_Yid);
    //printf("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n");
    //free(Y);//加一
//------------------------------------------------------------------------


//------------------------------------------------------------------------
/*
free(X);//加一
free(Y);//加一
free(Y_golden);//加一
free(csrSplitter);
free(Apinter);
free(Bpinter);
free(Yid);
free(Start1);
free(End1);
free(Ypartialsum);
free(Ysum);
free(Start2);
free(End2);
*/
}