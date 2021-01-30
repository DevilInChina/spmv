//
// Created by kouushou on 2020/12/10.
//
#include <spmv.h>
#if defined(__cplusplus)
extern "C" {
#endif
#ifndef GEMV_INNER_SPMV_H
#define GEMV_INNER_SPMV_H

/**
 * @brief create a empty handle with initialize
 * @return
 */
spmv_Handle_t gemv_create_handle();
/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param nnzR
 * @param nthreads
 */
void parallel_balanced_get_handle(
        spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        BASIC_INT_TYPE nnzR);

/**
 *
 * @param handle
 * @param m
 * @param RowPtr
 * @param nnzR
 * @param nthreads
 */
void parallel_balanced2_get_handle(
        spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE*RowPtr,
        BASIC_INT_TYPE nnzR);

void sell_C_Sigma_get_handle_Selected(spmv_Handle_t handle,
                                      BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE*RowPtr,
                                      const BASIC_INT_TYPE*ColIdx,
                                      const void*Matrix_Val
) ;

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size);
void init_csrSplitter_balanced2(int nthreads, int nnzR,
                                int m, const BASIC_INT_TYPE *RowPtr, BASIC_INT_TYPE *csrSplitter);

void init_csrSplitter_balanced(int nthreads, int nnzR,
                                int m, const BASIC_INT_TYPE *RowPtr, BASIC_INT_TYPE *csrSplitter);

void handle_init_common_parameters(spmv_Handle_t this_handle,
                                   BASIC_SIZE_TYPE nthreads,
                                   SPMV_METHODS function,
                                   BASIC_SIZE_TYPE size,
                                   VECTORIZED_WAY vectorizedWay);

void spmv_parallel_balanced_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y);

void spmv_parallel_balanced2_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y
);


void spmv_sell_C_Sigma_Selected(const spmv_Handle_t handle,
                                BASIC_INT_TYPE m,
                                const BASIC_INT_TYPE* RowPtr,
                                const BASIC_INT_TYPE* ColIdx,
                                const void* Matrix_Val,
                                const void* Vector_Val_X,
                                void*       Vector_Val_Y
);

void spmv_serial_Selected(const spmv_Handle_t handle,
                          BASIC_INT_TYPE m,
                          const BASIC_INT_TYPE* RowPtr,
                          const BASIC_INT_TYPE* ColIdx,
                          const void* Matrix_Val,
                          const void* Vector_Val_X,
                          void*       Vector_Val_Y);

void spmv_parallel_Selected(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE* RowPtr,
                            const BASIC_INT_TYPE* ColIdx,
                            const void* Matrix_Val,
                            const void* Vector_Val_X,
                            void*       Vector_Val_Y
);



void balancedHandleDestroy(spmv_Handle_t this_handle);

void balanced2HandleDestroy(spmv_Handle_t this_handle);



void csr5HandleDestory(spmv_Handle_t handle);
#ifdef NUMA
void numaHandleDestory(spmv_Handle_t handle);
int numa_spmv_get_handle_Selected(spmv_Handle_t handle,
                                  BASIC_INT_TYPE m,BASIC_INT_TYPE n,
                                  const BASIC_INT_TYPE *RowPtr,
                                  const BASIC_INT_TYPE *ColIdx,
                                  const void *Matrix_Val
);
void spmv_numa_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE* RowPtr,
        const BASIC_INT_TYPE* ColIdx,
        const void* Matrix_Val,
        const void* Vector_Val_X,
        void*       Vector_Val_Y
);
#endif
void csr5Spmv_get_handle_Selected(spmv_Handle_t handle,
                                  BASIC_INT_TYPE m,
                                  BASIC_INT_TYPE n,
                                  BASIC_INT_TYPE*RowPtr,
                                  BASIC_INT_TYPE*ColIdx,
                                  const void*Matrix_Val
) ;

void spmv_csr5Spmv_Selected(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE* RowPtr,
                            const BASIC_INT_TYPE* ColIdx,
                            const void* Matrix_Val,
                            const void* Vector_Val_X,
                            void*       Vector_Val_Y
);





typedef void(*spmv_function) (const spmv_Handle_t handle,
                      BASIC_INT_TYPE m,
                      const BASIC_INT_TYPE* RowPtr,
                      const BASIC_INT_TYPE* ColIdx,
                      const void* Matrix_Val,
                      const void* Vector_Val_X,
                      void*       Vector_Val_Y
);



extern const spmv_function spmv_functions[];

int lower_bound(const int *first,const int *last,int key);

int upper_bound(const int *first,const int *last,int key);


void inner_exclusive_scan(BASIC_INT_TYPE *input, int length);

void inner_matrix_transposition_d(const int           m,
                                  const int           n,
                                  const BASIC_INT_TYPE     nnz,
                                  const BASIC_INT_TYPE    *csrRowPtr,
                                  const int          *csrColIdx,
                                  const double *csrVal,
                                  int          *cscRowIdx,
                                  BASIC_INT_TYPE    *cscColPtr,
                                  double *cscVal);

void inner_matrix_transposition_s(const int           m,
                                  const int           n,
                                  const BASIC_INT_TYPE     nnz,
                                  const BASIC_INT_TYPE    *csrRowPtr,
                                  const int          *csrColIdx,
                                  const float *csrVal,
                                  int          *cscRowIdx,
                                  BASIC_INT_TYPE    *cscColPtr,
                                  float *cscVal);

void metis_partitioning(
        int m, int nnz,
        int nParts,
        int *RowPtr,
        int *ColIdx,
        int *part,
        void *val,BASIC_SIZE_TYPE size,const char *MtxToken);

void ReGather(void *true_val,const void*val ,const int *index,BASIC_SIZE_TYPE size,int len);

#endif //GEMV_INNER_SPMV_H

#if defined(__cplusplus)
}
#endif
