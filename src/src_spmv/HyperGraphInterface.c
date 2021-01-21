//
// Created by kouushou on 2021/1/21.
//
#include <stdlib.h>
#include "inner_spmv.h"
#include <metis.h>

void metis_partitioning(
        int n, int m, int nnz,
        int nParts,
        int *csrRowPtrA,
        int *csrColIdxA,
        void *val,BASIC_SIZE_TYPE size) {
    //int nn=n;
    //int nParts=2;
    int nn = n;
    //int nParts=2;
    int nWeights = 1;
    int *part = (int *) malloc(sizeof(int) * (nn + 1));
    int objval;
    int *csrRowPtrAAA = (int *) malloc(sizeof(int) * (nn + 1));
    int *csrColIdxAAA = (int *) malloc(sizeof(int) * nnz);
    for (int i = 0; i < m + 1; i++) {
        csrRowPtrAAA[i] = csrRowPtrA[i];
        //printf("csr=%ld \n",csrRowPtrAAA[i]);
    }

    for (int j = 0; j < nnz; j++) {
        csrColIdxAAA[j] = csrColIdxA[j];
        //printf("csr[%d]=%ld \n",csrColIdxAAA[j]);
    }
    int ret = METIS_PartGraphKway(&nn, &nWeights, csrRowPtrAAA, csrColIdxAAA,
                                  NULL, NULL, NULL, &nParts, NULL,
                                  NULL, NULL, &objval, part);

    int *outtxt = (int *) malloc(sizeof(int) * nn);
    for (int part_i = 0; part_i < nn; part_i++) {
        // printf("%d %ld\n",part_i,part[part_i]);
        outtxt[part_i] = part[part_i];
        //printf("%ld\n",outtxt[part_i]);
    }
    int colIdx_RSlen = 0;
    int *colIdx_RS = (int *) malloc(sizeof(int) * nnz);
    int *rowPtr_RS = (int *) malloc(sizeof(int) * (n + 1));
    //rowPtr_RS[0] = 0;
    int rp = -1;
    int temp = 0;
    for (int i = 0; i < nParts; i++) {
        for (int j = 0; j < n; j++) {
            if (outtxt[j] == i) {
                //printf("outtxt[%d] is %d\n", j, i);
                for (int q = csrRowPtrA[j]; q < csrRowPtrA[j + 1]; q++) {
                    colIdx_RS[colIdx_RSlen++] = csrColIdxA[q] + 1;
                    temp++;
                }
                //printf("temp=%d\n", temp);
                rowPtr_RS[++rp] = temp;
                temp = 0;
            }
        }
    }
    inner_exclusive_scan(rowPtr_RS, n + 1);
    //colnum sorting
    //csr to csc
    int *rowIdx_RS = (int *) malloc(sizeof(int) * nnz);
    int *colPtr_RS = (int *) malloc(sizeof(int) * (n + 1));
    void *val_RS = malloc(size * nnz);
    if(size==sizeof(double )){
        inner_matrix_transposition_d(n, n, nnz, rowPtr_RS, colIdx_RS, val,
                                     rowIdx_RS, colPtr_RS, val_RS);
    }else{
        inner_matrix_transposition_s(n, n, nnz, rowPtr_RS, colIdx_RS, val,
                                     rowIdx_RS, colPtr_RS, val_RS);
    }
    colPtr_RS[n] = nnz;

    int colIdx_CSlen = 0;
    int cp = 0;
    int *colIdx_RS_CS = (int *) malloc(sizeof(int) * nnz);

    for (int i = 0; i < nParts; i++) {
        for (int j = 0; j < n; j++) {
            if (outtxt[j] == i) {
                for (int q = colPtr_RS[j]; q < colPtr_RS[j + 1]; q++) {
                    colIdx_RS_CS[colIdx_CSlen++] = rowIdx_RS[q];
                }
            }
        }
    }
    for (int i = 0; i < m + 1; i++) {
        csrRowPtrA[i] = rowPtr_RS[i];
    }

    for (int j = 0; j < nnz; j++) {
        csrColIdxA[j] = colIdx_RS_CS[j];
    }
    free(part);
    free(csrColIdxAAA);
    free(csrRowPtrAAA);

    free(val_RS);
    free(colIdx_RS_CS);

    free(rowIdx_RS);
    free(colIdx_RS);

    free(colPtr_RS);
    free(colIdx_RS);

    free(outtxt);

}