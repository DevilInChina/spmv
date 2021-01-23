//
// Created by kouushou on 2021/1/21.
//
#include <stdlib.h>
#include "inner_spmv.h"
#include <metis.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct Pair {
    int first;
    int second;
} Pair, *Pair_t;

int cmp_pair(const void *A, const void *B) {
    const Pair_t Ac = A;
    const Pair_t Bc = B;
    int k = Ac->first - Bc->first;
    if (k)return k;
    else return Ac->second - Bc->second;
}

int cmp_int(const void *a, const void *b) {
    return *((int *) a) - *((int *) b);
}

void metis_partitioning(
        int m, int nnz,
        int nParts,
        int *RowPtr,
        int *ColIdx,
        int *part,
        void *val, BASIC_SIZE_TYPE size) {
    int nWeights = 1;
    int objval;
    int *cpyRowPtr = (int *) malloc(sizeof(int) * (m + 1));

    int *cpyColIdx = (int *) malloc(sizeof(int) * nnz);

    void *cpyVal = malloc(size * nnz);

    memcpy(cpyVal, val, size * nnz);

    memcpy(cpyRowPtr, RowPtr, sizeof(int) * (m + 1));

    memcpy(cpyColIdx, ColIdx, sizeof(int) * nnz);



    int ret = METIS_PartGraphKway(&m, &nWeights, cpyRowPtr, cpyColIdx,
                                  NULL, NULL, NULL, &nParts, NULL,
                                  NULL, NULL, &objval, part);
    Pair_t order = malloc(sizeof(Pair) * m);
    for (int i = 0; i < m; ++i) {
        order[i].first = part[i];
        order[i].second = i;
    }
    qsort(order,m,sizeof(Pair),cmp_pair);
    /// i put in order[i].second
    int nnzCount;
    cpyRowPtr[0] = 0;
    for (int i = 0; i < m; ++i) {
        cpyRowPtr[i + 1] = RowPtr[order[i].second + 1] - RowPtr[order[i].second];
        cpyRowPtr[i + 1] += cpyRowPtr[i];

    }

    for (int i = 0; i < m; ++i) {
        int len = cpyRowPtr[i + 1] - cpyRowPtr[i];

        memcpy(cpyColIdx + cpyRowPtr[i],
               ColIdx + RowPtr[order[i].second], sizeof(int) * len);

        memcpy(cpyVal + cpyRowPtr[i] * size,
               val + RowPtr[order[i].second] * size, size * len);

        part[i] = order[i].second;

    }

    memcpy(val, cpyVal, size * nnz);

    memcpy(RowPtr, cpyRowPtr, sizeof(int) * (m + 1));

    memcpy(ColIdx, cpyColIdx, sizeof(int) * (nnz));

    free(order);
    free(cpyVal);
    free(cpyColIdx);
    free(cpyRowPtr);
}

void ReGather(void *true_val, const void *val, int *index, BASIC_SIZE_TYPE size, int len) {
    //memcpy(true_val, val, size * len);
    if(size==sizeof(double )){
        double *tr = true_val;
        const double *ori = val;
        for (int i = 0; i < len; ++i) {
            tr[index[i]] = ori[i];
        }
    }else{
        float *tr = true_val;
        const float *ori = val;
        for (int i = 0; i < len; ++i) {
            tr[index[i]] = ori[i];
        }
    }
}