//
// Created by kouushou on 2020/11/25.
//
#include "inner_spmv.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

typedef struct balancedEnv {
    BASIC_INT_TYPE *csrSplitter;
    BASIC_INT_TYPE *Yid;
    BASIC_INT_TYPE *Apinter;
    BASIC_INT_TYPE *Start1;
    BASIC_INT_TYPE *End1;
    BASIC_INT_TYPE *Start2;
    BASIC_INT_TYPE *End2;
    BASIC_INT_TYPE *Bpinter;
    BASIC_INT_TYPE *label;
} balancedEnv, *balancedEnv_t;

void balanced2HandleDestroy(spmv_Handle_t this_handle) {
    if (this_handle) {
        if (this_handle->extraHandle && this_handle->spmvMethod == Method_Balanced2) {
            balancedEnv_t exh = (balancedEnv_t) this_handle->extraHandle;
            free(exh->csrSplitter);
            free(exh->Yid);
            free(exh->Apinter);
            free(exh->Start1);
            free(exh->End1);
            free(exh->Start2);
            free(exh->End2);
            free(exh->Bpinter);
            free(exh->label);
            free(exh);

            this_handle->extraHandle = NULL;
        }
    }
}

void init_csrSplitter_balanced2(int nthreads, int nnzR,
                                int m, const BASIC_INT_TYPE *RowPtr, BASIC_INT_TYPE *csrSplitter) {
    int stridennz = (nnzR + nthreads - 1) / nthreads;

    for (int tid = 0; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
    }
}

void parallel_balanced2_get_handle(
        spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        BASIC_INT_TYPE nnzR
) {
    int *csrSplitter = (int *) malloc(sizeof(int) * (handle->nthreads + 1));
    init_csrSplitter_balanced2((int) (handle->nthreads), nnzR, m, RowPtr, csrSplitter);

    BASIC_SIZE_TYPE nthreads = handle->nthreads;


    int *Yid = (int *) malloc(sizeof(int) * nthreads);
    memset(Yid, 0, sizeof(int) * nthreads);
    //每个线程
    int flag;
    int useBalanced = 1;
    for (int tid = 0; tid < nthreads; tid++) {
        //printf("tid = %i, csrSplitter: %i -> %i\n", tid, csrSplitter[tid], csrSplitter[tid+1]);
        if (csrSplitter[tid + 1] - csrSplitter[tid] == 0 && csrSplitter[tid] != m) {

            Yid[tid] = csrSplitter[tid];
            flag = 1;
            useBalanced = 0;
        } else {
            Yid[tid] = -1;
        }
        if (flag) {
            flag = 0;
        }
        //printf("Yid[%d] is %d\n", tid, Yid[tid]);
    }
    if (useBalanced) { //all of yid = -1
        handle->spmvMethod = Method_Balanced;
        handle->extraHandle = csrSplitter;
        free(Yid);
    } else {
        handle->spmvMethod = Method_Balanced2;
        int *Apinter = (int *) malloc(nthreads * sizeof(int));
        memset(Apinter, 0, nthreads * sizeof(int));
        //每个线程执行行数
        for (int tid = 0; tid < nthreads; tid++) {
            Apinter[tid] = csrSplitter[tid + 1] - csrSplitter[tid];
            //printf("A[%d] is %d\n", tid, Apinter[tid]);
        }

        int *Bpinter = (int *) malloc(nthreads * sizeof(int));
        memset(Bpinter, 0, nthreads * sizeof(int));

        for (int tid = 0; tid < nthreads; tid++) {
            int num = 0;
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                num += RowPtr[u + 1] - RowPtr[u];
            }
            Bpinter[tid] = num;
        }
        handle->extraHandle = malloc(sizeof(balancedEnv));
        balancedEnv_t Env = (balancedEnv_t) handle->extraHandle;
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
                int nntz = floor((double) Apinter[tid] / (double) (tid - start1 + 1));
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
                int nntz2 = floor((double) Bpinter[tid] / (double) (tid - start2 + 1));
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
        (Env)->Bpinter = Bpinter;
        (Env)->Apinter = Apinter;
        (Env)->Yid = Yid;
        (Env)->Start1 = Start1;
        (Env)->Start2 = Start2;
        (Env)->End1 = End1;
        (Env)->End2 = End2;
        (Env)->label = label;
        (Env)->csrSplitter = csrSplitter;
    }
}

 void spmv_parallel_balanced2_cpp_d(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const double *Matrix_Val,
        const double *Vector_Val_X,
        double *Vector_Val_Y
) {
    int nthreads = handle->nthreads;
    balancedEnv_t Env = (balancedEnv_t) handle->extraHandle;
    int *Yid = Env->Yid;
    int *csrSplitter = Env->csrSplitter;
    int *Apinter = Env->Apinter;
    int *Start1 = Env->Start1;
    int *Start2 = Env->Start2;
    int *End2 = Env->End2;
    int *End1 = Env->End1;
    int *label = Env->label;


    double *Ysum = (double *) malloc(sizeof(double) * nthreads);
    memset(Ysum, 0, sizeof(double) * nthreads);
    int cnt = 0;
    for (int tid = 0; tid < nthreads; tid++) {
        if (Yid[tid] != -1) {
            Vector_Val_Y[Yid[tid]] = 0;
        } else {
            ++cnt;
        }
    }
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++) {
        if (Yid[tid] == -1) {
            //printf("%d %d\n",tid,csrSplitter[tid]);
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                Dot_Product_Avx2_d(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u],
                                   Vector_Val_X,
                                   Vector_Val_Y + u);
            }
        } else if (label[tid] != 0) {

            for (int u = Start1[tid]; u < End1[tid]; u++) {
                Dot_Product_Avx2_d(
                        RowPtr[u + 1] - RowPtr[u],
                        ColIdx + RowPtr[u],
                        Matrix_Val + RowPtr[u],
                        Vector_Val_X,
                        Vector_Val_Y + u);
            }

        } else  {//if (Yid[tid] != -1 && label[tid] == 0)

            Dot_Product_Avx2_d(
                    End2[tid] - Start2[tid],
                    ColIdx + Start2[tid],
                    Matrix_Val + Start2[tid],
                    Vector_Val_X,
                    Ysum + tid
            );
            //((double *)Ysum)[tid] += ((double *)Ypartialsum)[tid];
            //CONVERT_ADDEQU(Vector_Val_Y+Yid[tid]*size,size,Ysum+tid*size);
        }
    }
    for (int tid = 0; tid < nthreads; ++tid) {
        if (Yid[tid] != -1) {
            Vector_Val_Y[Yid[tid]] += Ysum[tid];

        }
    }
    free(Ysum);
}

void spmv_parallel_balanced2_cpp_s(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const float *Matrix_Val,
        const float *Vector_Val_X,
        float *Vector_Val_Y
) {
    int nthreads = handle->nthreads;
    balancedEnv_t Env = (balancedEnv_t) handle->extraHandle;
    int *Yid = Env->Yid;
    int *csrSplitter = Env->csrSplitter;
    int *Apinter = Env->Apinter;
    int *Start1 = Env->Start1;
    int *Start2 = Env->Start2;
    int *End2 = Env->End2;
    int *End1 = Env->End1;
    int *label = Env->label;


    float *Ysum = (float *) malloc(sizeof(float) * nthreads);
    memset(Ysum, 0, sizeof(float) * nthreads);
    int cnt = 0;
    for (int tid = 0; tid < nthreads; tid++) {
        if (Yid[tid] != -1) {
            Vector_Val_Y[Yid[tid]] = 0;
        } else {
            ++cnt;
        }
    }
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++) {
        if (Yid[tid] == -1) {
            //printf("%d %d\n",tid,csrSplitter[tid]);
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                Dot_Product_Avx2_s(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u],
                                   Vector_Val_X,
                                   Vector_Val_Y + u);
            }
        } else if (label[tid] != 0) {

            for (int u = Start1[tid]; u < End1[tid]; u++) {
                Dot_Product_Avx2_s(
                        RowPtr[u + 1] - RowPtr[u],
                        ColIdx + RowPtr[u],
                        Matrix_Val + RowPtr[u],
                        Vector_Val_X,
                        Vector_Val_Y + u);
            }

        } else  {//if (Yid[tid] != -1 && label[tid] == 0)

            Dot_Product_Avx2_s(
                    End2[tid] - Start2[tid],
                    ColIdx + Start2[tid],
                    Matrix_Val + Start2[tid],
                    Vector_Val_X,
                    Ysum + tid
            );
            //((float *)Ysum)[tid] += ((float *)Ypartialsum)[tid];
            //CONVERT_ADDEQU(Vector_Val_Y+Yid[tid]*size,size,Ysum+tid*size);
        }
    }
    for (int tid = 0; tid < nthreads; ++tid) {
        if (Yid[tid] != -1) {
            Vector_Val_Y[Yid[tid]] += Ysum[tid];

        }
    }
    free(Ysum);
}
void spmv_parallel_balanced2_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y
) {
    if (handle->data_size == sizeof(double)) {
        spmv_parallel_balanced2_cpp_d(handle, m, RowPtr, ColIdx, (double *) Matrix_Val, (double *) Vector_Val_X,
                                      (double *) Vector_Val_Y);
    } else {
        spmv_parallel_balanced2_cpp_s(handle, m, RowPtr, ColIdx, (float *) Matrix_Val, (float *) Vector_Val_X,
                                      (float *) Vector_Val_Y);
    }
}



