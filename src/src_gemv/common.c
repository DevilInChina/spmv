//
// Created by kouushou on 2020/11/25.
//

#include <gemv.h>
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




const char*gemv_name[]={
        "parallel_balanced_gemv",
        "parallel_balanced_gemv_avx2",
        "parallel_balanced_gemv_avx512",
        "parallel_balanced2_gemv",
        "parallel_balanced2_gemv_avx2",
        "parallel_balanced2_gemv_avx512",
        "sell_C_Sigma_gemv",
        "sell_C_Sigma_gemv_avx2",
        "sell_C_Sigma_gemv_avx512",
};

const spmv_handle_function spmvs[]={
        NULL,
        spmv_parallel_balanced_Selected,
        spmv_parallel_balanced2_Selected,
        spmv_sell_C_Sigma_Selected,
};
