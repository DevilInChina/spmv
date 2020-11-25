//
// Created by kouushou on 2020/11/25.
//

#ifndef GEMV_COMMON_GEMV_H
#define GEMV_COMMON_GEMV_H
#include <gemv.h>
struct gemv_Handle {
    STATUS_GEMV_HANDLE status;
    GEMV_INT_TYPE nthreads;
    GEMV_INT_TYPE* csrSplitter;
    GEMV_INT_TYPE* Yid;
    GEMV_INT_TYPE* Apinter;
    GEMV_INT_TYPE* Start1;
    GEMV_INT_TYPE* End1;
    GEMV_INT_TYPE* Start2;
    GEMV_INT_TYPE* End2;
    GEMV_INT_TYPE* Bpinter;
};

#endif //GEMV_COMMON_GEMV_H
