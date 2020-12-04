//
// Created by kouushou on 2020/11/25.
//

#include <gemv.h>
#include <math.h>
#include <string.h>

void gemv_Handle_init(gemv_Handle_t this_handle){
    this_handle->status = NONE;
    this_handle->nthreads = 0;
    this_handle->csrSplitter = NULL;
    this_handle->Yid = NULL;
    this_handle->Apinter = NULL;
    this_handle->Start1 = NULL;
    this_handle->End1 = NULL;
    this_handle->Start2 = NULL;
    this_handle->End2 = NULL;
    this_handle->Bpinter = NULL;
}

void gemv_Handle_clear(gemv_Handle_t this_handle) {
    free(this_handle->csrSplitter);
    free(this_handle->Yid);
    free(this_handle->Apinter);
    free(this_handle->Start1);
    free(this_handle->End1);
    free(this_handle->Start2);
    free(this_handle->End2);
    free(this_handle->Bpinter);
    gemv_Handle_init(this_handle);
}

void gemv_destory_handle(gemv_Handle_t this_handle){
    gemv_Handle_clear(this_handle);
    free(this_handle);
}

gemv_Handle_t gemv_create_handle(){
    gemv_Handle_t ret = malloc(sizeof(gemv_Handle));
    gemv_Handle_init(ret);
    return ret;
}

void gemv_clear_handle(gemv_Handle_t this_handle){
    gemv_Handle_clear(this_handle);
}


int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}
#include <stdio.h>

void parallel_balanced_get_handle(
        gemv_Handle_t* handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE*RowPtr,
        GEMV_INT_TYPE nnzR,
        GEMV_INT_TYPE nthreads) {
    int *csrSplitter = (int *) malloc((nthreads + 1) * sizeof(int));
    //int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));

    int stridennz = ceil((double) nnzR / (double) nthreads);

#pragma omp parallel default(none) shared(nthreads, stridennz, nnzR, RowPtr, csrSplitter, m)
    for (int tid = 0; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
    }
    *handle = gemv_create_handle();
    (*handle)->nthreads = nthreads;
    (*handle)->status = BALANCED;
    (*handle)->csrSplitter = csrSplitter;

}
void parallel_balanced2_get_handle(
        gemv_Handle_t* handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE*RowPtr,
        GEMV_INT_TYPE nnzR,
        GEMV_INT_TYPE nthreads) {
    parallel_balanced_get_handle(handle, m, RowPtr, nnzR, nthreads);
    (*handle)->status = BALANCED2;
    int *csrSplitter = (*handle)->csrSplitter;

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
    (*handle)->Bpinter = Bpinter;
    (*handle)->Apinter = Apinter;
    (*handle)->Yid = Yid;
    (*handle)->Start1 = Start1;
    (*handle)->Start2 = Start2;
    (*handle)->Yid = Yid;
    (*handle)->End1 = End1;
    (*handle)->End2 = End2;
}

float hsum_avx(__m256 in256) {
    float sum;

    __m256 hsum = _mm256_hadd_ps(in256, in256);
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    _mm_store_ss(&sum, _mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));

    return sum;
}
float gemv_s_dotProduct(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X) {
    float ret = 0;
    for(int i = 0 ; i < len ; ++i){
        ret+=Val[i]*X[indx[i]];
    }
    return ret;
}

float gemv_s_dotProduct_avx2(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X) {
#ifdef DOT_AVX2_CAN
    float sum = 0;
    __m256 res = _mm256_setzero_ps();
    const int DEPTH = 8;
    int dif = len;
    int nloop = dif / DEPTH;
    int remainder = dif % DEPTH;
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH) {

        __m256 vecv = _mm256_loadu_ps(&Val[j]);
        __m256i veci = _mm256_loadu_si256((__m256i *) (&indx[j]));
        __m256 vecx = _mm256_i32gather_ps(X, veci, 4);
        res = _mm256_fmadd_ps(vecv, vecx, res);
    }
    //Y[u] += _mm256_reduce_add_ps(res);
    sum += hsum_avx(res);

    for (int j = nloop * DEPTH; j < len; ++j) {
        sum += Val[j] * X[indx[j]];
    }
    return sum;
#else
    return gemv_s_dotProduct(len,indx,Val,X);
#endif
}


float gemv_s_dotProduct_avx512(
        GEMV_INT_TYPE len,const GEMV_INT_TYPE* indx,const float *Val,const float *X){
#ifdef DOT_AVX512_CAN
    float sum = 0;
    __m512 res = _mm512_setzero_ps();
    int dif = len;
    const int DEPTH=16;
    int nloop = dif / DEPTH;
    int remainder = dif % DEPTH;
    for (int li = 0,j = 0; li < nloop; li++,j+=DEPTH)
    {
        __m512 vecv = _mm512_loadu_ps(&Val[j]);
        __m512i veci =  _mm512_loadu_si512(&indx[j]);
        __m512 vecx = _mm512_i32gather_ps (veci, X, 4);
        res = _mm512_fmadd_ps(vecv, vecx, res);
    }
    sum += _mm512_reduce_add_ps(res);

    for (int j = nloop * DEPTH; j < len; j++) {
        sum += Val[j] * X[indx[j]];
    }
    return sum;
#else
    return gemv_s_dotProduct(len,indx,Val,X);
#endif
}

void parallel_balanced_gemv_Selected(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y,
        DOT_PRODUCT_WAY way
) {
    if(handle->status!=BALANCED) {
        return;
    }
    float (*dot_product)(GEMV_INT_TYPE len, const GEMV_INT_TYPE *indx, const float *Val, const float *X);
    switch (way) {
        case DOT_AVX2: {
            dot_product = gemv_s_dotProduct_avx2;
        }
            break;
        case DOT_AVX512: {
            dot_product = gemv_s_dotProduct_avx512;
        }
            break;
        default: {
            dot_product = gemv_s_dotProduct;
        }
            break;
    }

    const int *csrSplitter = handle->csrSplitter;
    const int nthreads = handle->nthreads;
    {
#pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++) {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                Vector_Val_Y[u] = dot_product(
                        RowPtr[u + 1] - RowPtr[u], ColIdx + RowPtr[u],
                        Matrix_Val + RowPtr[u], Vector_Val_X);
            }
        }
    }
}



void parallel_balanced2_gemv_Selected(
        const gemv_Handle_t handle,
        GEMV_INT_TYPE m,
        const GEMV_INT_TYPE* RowPtr,
        const GEMV_INT_TYPE* ColIdx,
        const GEMV_VAL_TYPE* Matrix_Val,
        const GEMV_VAL_TYPE* Vector_Val_X,
        GEMV_VAL_TYPE*       Vector_Val_Y,
        DOT_PRODUCT_WAY way
) {
    if(handle->status!=BALANCED2) {
        return;
    }
    int nthreads = handle->nthreads;
    int *Yid = handle->Yid;
    int *csrSplitter = handle->csrSplitter;
    int *Apinter = handle->Apinter;
    int *Start1 = handle->Start1;
    int *Start2 = handle->Start2;
    int *End2 = handle->End2;
    int *End1 = handle->End1;
    float (*dot_product)(GEMV_INT_TYPE len, const GEMV_INT_TYPE *indx, const float *Val, const float *X);
    switch (way) {
        case DOT_AVX2: {
            dot_product = gemv_s_dotProduct_avx2;
        }
            break;
        case DOT_AVX512: {
            dot_product = gemv_s_dotProduct_avx512;
        }
            break;
        default: {
            dot_product = gemv_s_dotProduct;
        }
            break;
    }
    {
#pragma omp parallel default(shared)
        for (int tid = 0; tid < nthreads; tid++)
            Vector_Val_Y[Yid[tid]] = 0;
        int *Ysum = malloc(sizeof(int) * nthreads);
        int *Ypartialsum = malloc(sizeof(int) * nthreads);
#pragma omp parallel default(shared)
        for (int tid = 0; tid < nthreads; tid++) {
            if (Yid[tid] == -1) {
                for (int u = csrSplitter[tid]; u < csrSplitter[tid + 1]; u++) {
                    Vector_Val_Y[u] =
                            dot_product(RowPtr[u+1]-RowPtr[u],
                                        ColIdx+RowPtr[u],
                                        Matrix_Val+RowPtr[u],Vector_Val_X);
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] > 1) {
                for (int u = Start1[tid]; u < End1[tid]; u++) {
                    Vector_Val_Y[u] =
                            dot_product(RowPtr[u+1]-RowPtr[u],
                                        ColIdx+RowPtr[u],
                                        Matrix_Val+RowPtr[u],Vector_Val_X);
                }
            }
            if (Yid[tid] != -1 && Apinter[tid] <= 1) {
                Ysum[tid] = 0;
                Ypartialsum[tid] = dot_product(End2[tid]-Start2[tid],
                                               ColIdx+Start2[tid],Matrix_Val+Start2[tid],Vector_Val_X);
                Ysum[tid] += Ypartialsum[tid];
                Vector_Val_Y[Yid[tid]] += Ysum[tid];
            }
        }
        free(Ysum);
        free(Ypartialsum);
    }
}
