//
// Created by kouushou on 2020/11/25.
//
#include "inner_spmv.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

void init_csrSplitter_balanced2(int nthreads,int nnzR,
                                int m,const BASIC_INT_TYPE*RowPtr,BASIC_INT_TYPE *csrSplitter){
    int stridennz = (nnzR+nthreads-1) /  nthreads;

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

    (handle)->csrSplitter = malloc(sizeof(int) * (handle->nthreads + 1));
    init_csrSplitter_balanced2((int) (handle->nthreads), nnzR, m, RowPtr, handle->csrSplitter);

    int *csrSplitter = (handle)->csrSplitter;
    BASIC_SIZE_TYPE nthreads = handle->nthreads;

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
            int nntz = floor((double ) Apinter[tid] / (double) (tid - start1 + 1));
            if (nntz != 0) {
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

    //非零元平均用在一行
    float *Ypartialsum = (float *) malloc(sizeof(float) * nthreads);
    memset(Ypartialsum, 0, sizeof(float) * nthreads);
    float *Ysum = (float *) malloc(sizeof(float) * nthreads);
    memset(Ysum, 0, sizeof(float) * nthreads);
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
            int nntz2 = ceil((float) Bpinter[tid] / (float) (tid - start2 + 1));
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
    (handle)->Bpinter = Bpinter;
    (handle)->Apinter = Apinter;
    (handle)->Yid = Yid;
    (handle)->Start1 = Start1;
    (handle)->Start2 = Start2;
    (handle)->Yid = Yid;
    (handle)->End1 = End1;
    (handle)->End2 = End2;
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
    int nthreads = handle->nthreads;
    int *Yid = handle->Yid;
    int *csrSplitter = handle->csrSplitter;
    int *Apinter = handle->Apinter;
    int *Start1 = handle->Start1;
    int *Start2 = handle->Start2;
    int *End2 = handle->End2;
    int *End1 = handle->End1;
    BASIC_SIZE_TYPE size = handle->data_size;
    VECTORIZED_WAY way = handle->vectorizedWay;
    dot_product_function dotProductFunction = inner_basic_GetDotProduct(size);



    void *Ysum = malloc(size * nthreads);
    void *Ypartialsum = malloc(size * nthreads);
    memset(Ysum,0,size*nthreads);
    memset(Ypartialsum,0,size*nthreads);
    for (int tid = 0; tid < nthreads; tid++) {
        if(Yid[tid]!=-1) {
            CONVERT_EQU(Vector_Val_Y+Yid[tid]*size,size,0);
        }
    }
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++) {
        if (Yid[tid] == -1) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                dotProductFunction(RowPtr[u + 1] - RowPtr[u],
                                   ColIdx + RowPtr[u],
                                   Matrix_Val + RowPtr[u]*size, Vector_Val_X,Vector_Val_Y+u*size,way);
            }
        }
        else if (Yid[tid] != -1 && Apinter[tid] > 1) {
            for (int u = Start1[tid]; u < End1[tid]; u++) {
                dotProductFunction(
                        RowPtr[u + 1] - RowPtr[u],
                        ColIdx + RowPtr[u],
                        Matrix_Val + RowPtr[u]*size, Vector_Val_X,Vector_Val_Y+u*size,way);
            }
        }
        else if (Yid[tid] != -1 && Apinter[tid] <= 1) {

            dotProductFunction(
                    End2[tid] - Start2[tid],
                    ColIdx + Start2[tid], Matrix_Val + Start2[tid]*size, Vector_Val_X,
                    Ypartialsum+tid*size,way
            );
            //((double *)Ysum)[tid] += ((double *)Ypartialsum)[tid];
            CONVERT_ADDEQU(Ysum+tid*size,size,Ypartialsum+tid*size);

            //CONVERT_ADDEQU(Vector_Val_Y+Yid[tid]*size,size,Ysum+tid*size);
        }
    }
    for(int tid = 0 ; tid < nthreads ; ++tid){
        if(Yid[tid]!=-1) {
            CONVERT_ADDEQU(Vector_Val_Y+Yid[tid]*size,size,Ysum+tid*size);
        }
    }


    free(Ysum);
    free(Ypartialsum);
}



