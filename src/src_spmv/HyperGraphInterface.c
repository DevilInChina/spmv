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
    void *val;
} Pair, *Pair_t;

int cmp_pair(const void *A, const void *B) {
    const Pair_t *Ac = A;
    const Pair_t *Bc = B;
    int k = (*Ac)->first - (*Bc)->first;
    if (k)return k;
    else return (*Ac)->second - (*Bc)->second;
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

    int *revIndex = (int *) malloc(sizeof(int) * m);

    void *cpyVal = malloc(size * nnz);

    memcpy(cpyVal, val, size * nnz);

    memcpy(cpyRowPtr, RowPtr, sizeof(int) * (m + 1));

    memcpy(cpyColIdx, ColIdx, sizeof(int) * nnz);


    int ret = METIS_PartGraphKway(&m, &nWeights, cpyRowPtr, cpyColIdx,
                                  NULL, NULL, NULL, &nParts, NULL,
                                  NULL, NULL, &objval, part);
    Pair_t *order = malloc(sizeof(Pair_t) * m);
    for (int i = 0; i < m; ++i) {
        order[i] = malloc(sizeof(struct Pair));
        order[i]->first = part[i];
        order[i]->second = i;
    }
    qsort(order, m, sizeof(Pair_t), cmp_pair);
    for (int i = 0; i < m; ++i) {
        part[i] = order[i]->second;
        revIndex[part[i]] = i;
    }
    for(int i = 0 ; i < m ; ++i) {
        //printf("%d %d\n", revIndex[i],part[i]);
    }
    /// i put in order[i].second
    int nnzCount;
    cpyRowPtr[0] = 0;
    for (int i = 0; i < m; ++i) {
        cpyRowPtr[i + 1] = RowPtr[part[i] + 1] - RowPtr[part[i]];
        cpyRowPtr[i + 1] += cpyRowPtr[i];
    }

    for (int i = 0; i < m; ++i) {
        int len = cpyRowPtr[i + 1] - cpyRowPtr[i];

        for (int j = RowPtr[part[i]]; j < RowPtr[part[i] + 1]; ++j) {
            order[j - RowPtr[part[i]]]->first = part[ColIdx[j]];
            order[j - RowPtr[part[i]]]->second = ColIdx[j];
            order[j - RowPtr[part[i]]]->val = val + size * j;
        }
        qsort(order, len, sizeof(Pair_t), cmp_pair);
        for (int j = 0; j < len; ++j) {
            cpyColIdx[j + cpyRowPtr[i]] = order[j]->second;
            memcpy(cpyVal + (j + cpyRowPtr[i]) * size, order[j]->val, size);
        }
    }


    memcpy(val, cpyVal, size * nnz);

    memcpy(RowPtr, cpyRowPtr, sizeof(int) * (m + 1));

    memcpy(ColIdx, cpyColIdx, sizeof(int) * (nnz));
    for(int j = RowPtr[0] ; j < RowPtr[1] ; ++j){
        //printf("%d %d\n",part[ColIdx[j]],ColIdx[j]);
    }
    for (int i = 0; i < m; ++i)free(order[i]);
    free(order);
    free(cpyVal);
    free(cpyColIdx);
    free(cpyRowPtr);
    free(revIndex);
}

void ReGather(void *true_val, const void *val, int *index, BASIC_SIZE_TYPE size, int len) {
    //memcpy(true_val, val, size * len);
    if (size == sizeof(double)) {
        double *tr = true_val;
        const double *ori = val;
        for (int i = 0; i < len; ++i) {
            tr[index[i]] = ori[i];
        }
    } else {
        float *tr = true_val;
        const float *ori = val;
        for (int i = 0; i < len; ++i) {
            tr[index[i]] = ori[i];
        }
    }
}