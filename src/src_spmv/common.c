//
// Created by kouushou on 2020/11/25.
//

#include <inner_spmv.h>
#include <math.h>
#include <string.h>
/**
 * @brief init parameters used in balanced and balanced2
 * @param this_handle
 */
void init_Balance_Balance2(gemv_Handle_t this_handle){
    this_handle->csrSplitter = NULL;
    this_handle->Yid = NULL;
    this_handle->Apinter = NULL;
    this_handle->Start1 = NULL;
    this_handle->End1 = NULL;
    this_handle->Start2 = NULL;
    this_handle->End2 = NULL;
    this_handle->Bpinter = NULL;
}

void init_sell_C_Sigma(gemv_Handle_t this_handle){
    this_handle->Sigma = 0;
    this_handle->C = 0;
    this_handle->banner = 0;

    this_handle->C_Blocks = NULL;
}

/**
 * @brief free parameters used in balanced and balanced2
 * @param this_handle
 */
void clear_Balance_Balance2(gemv_Handle_t this_handle){
    free(this_handle->csrSplitter);
    free(this_handle->Yid);
    free(this_handle->Apinter);
    free(this_handle->Start1);
    free(this_handle->End1);
    free(this_handle->Start2);
    free(this_handle->End2);
    free(this_handle->Bpinter);
}

void C_Block_destory(C_Block_t this_block){
    free(this_block->RowIndex);
    free(this_block->ColIndex);
    free(this_block->ValT);
    free(this_block->Y);
}

void clear_Sell_C_Sigma(gemv_Handle_t this_handle) {
    int siz = this_handle->banner / (this_handle->C ? this_handle->C : 1);
    for (int i = 0; i < siz; ++i) {
        C_Block_destory(this_handle->C_Blocks + i);
    }
    free(this_handle->C_Blocks);
}

void gemv_Handle_init(gemv_Handle_t this_handle){
    this_handle->status = STATUS_NONE;
    this_handle->nthreads = 0;

    init_Balance_Balance2(this_handle);
    init_sell_C_Sigma(this_handle);

}

void gemv_Handle_clear(gemv_Handle_t this_handle) {
    clear_Balance_Balance2(this_handle);
    clear_Sell_C_Sigma(this_handle);

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






void handle_init_common_parameters(gemv_Handle_t this_handle,
                                   BASIC_SIZE_TYPE nthreads,
                                   STATUS_GEMV_HANDLE function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay){
    this_handle->nthreads = nthreads;
    this_handle->vectorizedWay = vectorizedWay;
    this_handle->data_size = size;
    this_handle->status = function;
}

const spmv_function spmv_functions[] = {
        spmv_serial_Selected,
        spmv_parallel_Selected,
        spmv_parallel_balanced_Selected,
        spmv_parallel_balanced2_Selected,
        spmv_sell_C_Sigma_Selected
};

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

void parallel_balanced_get_handle(
        gemv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        BASIC_INT_TYPE nnzR
) {
    BASIC_SIZE_TYPE nthreads = handle->nthreads;
    int *csrSplitter = (int *) malloc((nthreads + 1) * sizeof(int));
    //int *csrSplitter_normal = (int *)malloc((nthreads+1) * sizeof(int));

    int stridennz = (nnzR+nthreads-1)/nthreads;

#pragma omp parallel default(none) shared(nthreads, stridennz, nnzR, RowPtr, csrSplitter, m)
    for (int tid = 0; tid <= nthreads; tid++) {
        // compute partition boundaries by partition of size stride
        int boundary = tid * stridennz;
        // clamp partition boundaries to [0, nnzR]
        boundary = boundary > nnzR ? nnzR : boundary;
        // binary search
        csrSplitter[tid] = binary_search_right_boundary_kernel(RowPtr, boundary, m + 1) - 1;
    }
    (handle)->csrSplitter = csrSplitter;

}

void parallel_balanced2_get_handle(
        gemv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        BASIC_INT_TYPE nnzR
) {

    parallel_balanced_get_handle(handle, m, RowPtr, nnzR);
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
    BASIC_SIZE_TYPE sto = nthreads > nnzR ? nthreads : nnzR;
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
    (handle)->Bpinter = Bpinter;
    (handle)->Apinter = Apinter;
    (handle)->Yid = Yid;
    (handle)->Start1 = Start1;
    (handle)->Start2 = Start2;
    (handle)->Yid = Yid;
    (handle)->End1 = End1;
    (handle)->End2 = End2;
}



void spmv_create_handle_all_in_one(gemv_Handle_t *Handle,
                                   BASIC_INT_TYPE m,
                                   const BASIC_INT_TYPE*RowPtr,
                                   const BASIC_INT_TYPE *ColIdx,
                                   const void *Matrix_Val,
                                   BASIC_SIZE_TYPE nthreads,
                                   STATUS_GEMV_HANDLE Function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay
){
    *Handle = gemv_create_handle();
    if(Function<STATUS_NONE || Function>=STATUS_TOTAL_SIZE)Function = STATUS_NONE;

    handle_init_common_parameters(*Handle,nthreads,Function,size,vectorizedWay);

    switch (Function) {
        case STATUS_BALANCED:{
            parallel_balanced_get_handle(*Handle,m,RowPtr,RowPtr[m]-RowPtr[0]);
        }break;
        case STATUS_BALANCED2:{
            parallel_balanced2_get_handle(*Handle,m,RowPtr,RowPtr[m]-RowPtr[0]);
        }break;
        case STATUS_SELL_C_SIGMA:{
            sell_C_Sigma_get_handle_Selected(*Handle,4,32,m,RowPtr,ColIdx,Matrix_Val);
        }
        default:{

            return;
        }
    }
}

void spmv(const gemv_Handle_t handle,
          BASIC_INT_TYPE m,
          const BASIC_INT_TYPE* RowPtr,
          const BASIC_INT_TYPE* ColIdx,
          const void* Matrix_Val,
          const void* Vector_Val_X,
          void*       Vector_Val_Y){
    if(handle==NULL)return;
    spmv_functions[handle->status](handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y);
}

#define STR(args1,args2) #args1 #args2

#define VEC_STRING(NAME)\
STR(NAME,_VECTOR_NONE),\
STR(NAME,_VECTOR_AVX2),\
STR(NAME,_VECTOR_AVX512)

#define ALL_FUNC_SRTING \
VEC_STRING(STATUS_NONE),\
VEC_STRING(STATUS_PARALLEL),\
VEC_STRING(STATUS_BALANCED),\
VEC_STRING(STATUS_BALANCED2),\
VEC_STRING(STATUS_SELL_C_SIGMA)

const char * funcNames[]= {
    ALL_FUNC_SRTING
};