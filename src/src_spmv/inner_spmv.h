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
        const BASIC_INT_TYPE *RowPtr,
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
        const BASIC_INT_TYPE *RowPtr,
        BASIC_INT_TYPE nnzR);

void sell_C_Sigma_get_handle_Selected(spmv_Handle_t handle,
                                      BASIC_INT_TYPE Times, BASIC_INT_TYPE C,
                                      BASIC_INT_TYPE m,
                                      const BASIC_INT_TYPE *RowPtr,
                                      const BASIC_INT_TYPE *ColIdx,
                                      const void *Matrix_Val
);

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int key_input,
                                        const int size);

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
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y);

void spmv_parallel_balanced2_Selected(
        const spmv_Handle_t handle,
        BASIC_INT_TYPE m,
        const BASIC_INT_TYPE *RowPtr,
        const BASIC_INT_TYPE *ColIdx,
        const void *Matrix_Val,
        const void *Vector_Val_X,
        void *Vector_Val_Y
);


void spmv_sell_C_Sigma_Selected(const spmv_Handle_t handle,
                                BASIC_INT_TYPE m,
                                const BASIC_INT_TYPE *RowPtr,
                                const BASIC_INT_TYPE *ColIdx,
                                const void *Matrix_Val,
                                const void *Vector_Val_X,
                                void *Vector_Val_Y
);

void spmv_serial_Selected(const spmv_Handle_t handle,
                          BASIC_INT_TYPE m,
                          const BASIC_INT_TYPE *RowPtr,
                          const BASIC_INT_TYPE *ColIdx,
                          const void *Matrix_Val,
                          const void *Vector_Val_X,
                          void *Vector_Val_Y);

void spmv_parallel_Selected(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE *RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const void *Matrix_Val,
                            const void *Vector_Val_X,
                            void *Vector_Val_Y
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
                                  BASIC_INT_TYPE *RowPtr,
                                  BASIC_INT_TYPE *ColIdx,
                                  const void *Matrix_Val
);

void spmv_csr5Spmv_Selected(const spmv_Handle_t handle,
                            BASIC_INT_TYPE m,
                            const BASIC_INT_TYPE *RowPtr,
                            const BASIC_INT_TYPE *ColIdx,
                            const void *Matrix_Val,
                            const void *Vector_Val_X,
                            void *Vector_Val_Y
);


typedef void(*spmv_function)(const spmv_Handle_t handle,
                             BASIC_INT_TYPE m,
                             const BASIC_INT_TYPE *RowPtr,
                             const BASIC_INT_TYPE *ColIdx,
                             const void *Matrix_Val,
                             const void *Vector_Val_X,
                             void *Vector_Val_Y
);


extern const spmv_function spmv_functions[];

int lower_bound(const int *first, const int *last, int key);

int upper_bound(const int *first, const int *last, int key);


void inner_exclusive_scan(BASIC_INT_TYPE *input, int length);

void inner_matrix_transposition_d(const int m,
                                  const int n,
                                  const BASIC_INT_TYPE nnz,
                                  const BASIC_INT_TYPE *csrRowPtr,
                                  const int *csrColIdx,
                                  const double *csrVal,
                                  int *cscRowIdx,
                                  BASIC_INT_TYPE *cscColPtr,
                                  double *cscVal);

void inner_matrix_transposition_s(const int m,
                                  const int n,
                                  const BASIC_INT_TYPE nnz,
                                  const BASIC_INT_TYPE *csrRowPtr,
                                  const int *csrColIdx,
                                  const float *csrVal,
                                  int *cscRowIdx,
                                  BASIC_INT_TYPE *cscColPtr,
                                  float *cscVal);

#if (OPT_LEVEL == 3)

void metis_partitioning(
        int m, int nnz,
        int nParts,
        int *RowPtr,
        int *ColIdx,
        int *part,
        void *val, BASIC_SIZE_TYPE size, const char *MtxToken);

void ReGather(void *true_val, const void *val, const int *index, BASIC_SIZE_TYPE size, int len);

#endif

inline void Dot_Product_Avx2_d(BASIC_INT_TYPE len,
                        const BASIC_INT_TYPE *indx,
                        const double *Val,
                        const double *X,
                        double *res) {

    const BASIC_INT_TYPE *colIndPtr = indx;
    const double *matValPtr = (double *) Val;
    const double *x = (double *) X;

    BASIC_INT_TYPE j;
    double result = 0.0;

    __m256d vec_y;
    vec_y = _mm256_setzero_pd();
    BASIC_INT_TYPE nnzThisLine = len;
    BASIC_INT_TYPE k_iter = nnzThisLine / 4;
    BASIC_INT_TYPE k_rem = nnzThisLine % 4;

    //Loop in multiples of 4 non-zeroes
    for (j = 0; j < k_iter; j++) {
        vec_y = _mm256_fmadd_pd(
                *((__m256d_u *) (matValPtr)),
                _mm256_set_pd(x[*(colIndPtr + 3)],
                              x[*(colIndPtr + 2)],
                              x[*(colIndPtr + 1)],
                              x[*(colIndPtr)]),
                vec_y);

        matValPtr += 4;
        colIndPtr += 4;
    }

    // Horizontal addition
    if (k_iter) {
        // sum[0] += sum[1] ; sum[2] += sum[3]
        vec_y = _mm256_hadd_pd(vec_y, vec_y);
        // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
        __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
        // Extract 128 bits to obtain sum[2] and sum[3]
        __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
        // Add remaining two sums
        __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
        // Store result
        result = sse_sum[0];
    }

    //Remainder loop for nnzThisLine%4
    for (j = 0; j < k_rem; j++) {
        result += *matValPtr++ * x[*colIndPtr++];
    }


    *(double *) res = result;
}

inline void Dot_Product_Avx2_s(BASIC_INT_TYPE len,
                        const BASIC_INT_TYPE *indx,
                        const float *Val,
                        const float *X,
                        float *res) {
    BASIC_INT_TYPE j;
    float result = 0.0;
    __m256 vec_y = _mm256_setzero_ps();
    BASIC_INT_TYPE nnz = len;
    BASIC_INT_TYPE k_iter = nnz / 8;
    BASIC_INT_TYPE k_rem = nnz % 8;

    const BASIC_INT_TYPE *colIndPtr = indx;
    const float *matValPtr = (float *) Val;
    const float *x = (float *) X;

    //Loop in multiples of 8
    for (j = 0; j < k_iter; j++) {

        vec_y = _mm256_fmadd_ps(
                _mm256_loadu_ps(matValPtr),
                _mm256_set_ps(x[*(colIndPtr + 7)],
                              x[*(colIndPtr + 6)],
                              x[*(colIndPtr + 5)],
                              x[*(colIndPtr + 4)],
                              x[*(colIndPtr + 3)],
                              x[*(colIndPtr + 2)],
                              x[*(colIndPtr + 1)],
                              x[*(colIndPtr)]),
                vec_y);

        matValPtr += 8;
        colIndPtr += 8;
    }

    // Horizontal addition of vec_y
    if (k_iter) {
        // hiQuad = ( x7, x6, x5, x4 )
        __m128 hiQuad = _mm256_extractf128_ps(vec_y, 1);
        // loQuad = ( x3, x2, x1, x0 )
        const __m128 loQuad = _mm256_castps256_ps128(vec_y);
        // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
        const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
        // loDual = ( -, -, x1 + x5, x0 + x4 )
        const __m128 loDual = sumQuad;
        // hiDual = ( -, -, x3 + x7, x2 + x6 )
        const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
        // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
        const __m128 sumDual = _mm_add_ps(loDual, hiDual);
        // lo = ( -, -, -, x0 + x2 + x4 + x6 )
        const __m128 lo = sumDual;
        // hi = ( -, -, -, x1 + x3 + x5 + x7 )
        const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
        // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
        const __m128 sum = _mm_add_ss(lo, hi);
        result = _mm_cvtss_f32(sum);
    }

    //Remainder loop
    for (j = 0; j < k_rem; j++) {
        result += *matValPtr++ * x[*colIndPtr++];
    }

    // Perform alpha * A * x

    *res = result;
}

#define LINE_S_PRODUCTGather_PARAMETERS_IN const BASIC_INT_TYPE ld, const BASIC_INT_TYPE length, const float*Val,const BASIC_INT_TYPE* indx, const float *Vector_X,const BASIC_INT_TYPE*indy,float *Vector_Y
#define LINE_D_PRODUCTGather_PARAMETERS_IN const BASIC_INT_TYPE ld, const BASIC_INT_TYPE length, const double*Val,const BASIC_INT_TYPE* indx, const double *Vector_X,const BASIC_INT_TYPE*indy,double *Vector_Y
#define LINE_PRODUCTGather_PARAMETERS_CALL(banner) Val+(banner),indx+(banner),Vector_X,indy+(banner),Vector_Y


inline void basic_s_lineProductGather_len(LINE_S_PRODUCTGather_PARAMETERS_IN) {
    for (int i = 0; i < ld; ++i) {
        for (int j = 0; j < length; ++j) {
            if (i)
                Vector_Y[indy[j]] += Val[i * length + j] * Vector_X[indx[i * length + j]];
            else Vector_Y[indy[j]] = Val[i * length + j] * Vector_X[indx[i * length + j]];
        }
    }
}


inline void basic_d_lineProductGather_len(LINE_D_PRODUCTGather_PARAMETERS_IN) {
    const int block = 8;
    for(int i = 0 ; i < length ; i+=block){
        double vec[8] = {0.0};
        for(int j = 0 ; j < ld ; ++j){
            for(int k = 0 ; k < block ; ++k){
                vec[k]+=Val[j*length+i+k]*Vector_X[indx[j*length+i+k]];
            }
        }
        for(int k = 0 ; k < block ; ++k){
            Vector_Y[indy[k+i]]= vec[k];
        }
    }
}

inline void basic_s_lineProductGather_avx2(LINE_S_PRODUCTGather_PARAMETERS_IN) {
    const int block = 8;
    for (int i = 0; i < length; i += block) {
        __m256_u vecy = _mm256_setzero_ps();
        const float *ValLine = Val + i;
        const int *indxLine = indx + i;

        for (int j = 0; j < ld; ++j) {
            vecy = _mm256_fmadd_ps(
                    *(__m256_u *) (ValLine),
                    _mm256_set_ps(Vector_X[*(indxLine + 7)],
                                  Vector_X[*(indxLine + 6)],
                                  Vector_X[*(indxLine + 5)],
                                  Vector_X[*(indxLine + 4)],
                                  Vector_X[*(indxLine + 3)],
                                  Vector_X[*(indxLine + 2)],
                                  Vector_X[*(indxLine + 1)],
                                  Vector_X[*(indxLine)]), vecy
            );
            ValLine+=length;
            indxLine+=length;
        }
        float *cur = (float *) (&vecy);

        for (int j = 0; j < block; ++j) {
            Vector_Y[indy[i + j]] = cur[j];
        }
    }
}

inline void basic_d_lineProductGather_avx2(LINE_D_PRODUCTGather_PARAMETERS_IN) {
    const int block = 4;
    for (int i = 0; i < length; i += block) {
        __m256d_u vecy = _mm256_setzero_pd();
        const double *ValLine = Val + i;
        const int *indxLine = indx + i;
        for (int j = 0; j < ld; ++j) {

            vecy = _mm256_fmadd_pd(
                    *(__m256d_u *) (ValLine),
                    _mm256_set_pd(Vector_X[*(indxLine + 3)],
                                  Vector_X[*(indxLine + 2)],
                                  Vector_X[*(indxLine + 1)],
                                  Vector_X[*(indxLine)]), vecy
            );
            ValLine+=length;
            indxLine+=length;
        }
        double *cur = (double *) (&vecy);
        for (int j = 0; j < block; ++j) {
            Vector_Y[indy[i + j]] = cur[j];
        }
    }
}



#endif //GEMV_INNER_SPMV_H

#if defined(__cplusplus)
}
#endif
