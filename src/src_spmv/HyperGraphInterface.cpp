//
// Created by kouushou on 2021/1/21.
//
#if (OPT_LEVEL==3)
#include "inner_spmv.h"
#include <metis.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;



int cmp_int(const void *a, const void *b) {
    return *((int *) a) - *((int *) b);
}


bool findToken(int m,int nnz,const char *MtxToken,int *part){
    string fileName(MtxToken);
    for(auto &it:fileName){
        if(it==' ' || it == '/' || it == '\\')it='_';
    }
    fileName = "cache/" + fileName;
    fileName+=".bin";
    ifstream inFile(fileName, ios::in | ios::binary);
    if(!inFile.is_open())return false;
    int m_R,nnz_R;
    inFile.read((char*)(&m_R),sizeof(int ));

    if(m_R!=m || inFile.eof())return false;

    inFile.read((char*)(&nnz_R),sizeof(int ));
    if(nnz_R!=nnz || inFile.eof())return false;

    inFile.read((char*)(part),sizeof(int )*m);
    inFile.close();
    return true;
}
void loadToken(int m,int nnz,const char *MtxToken,const int *part){
    string fileName(MtxToken);
    for(auto &it:fileName){
        if(it==' ' || it == '/' || it == '\\')it='_';
    }
    fileName = "cache/" + fileName;
    fileName+=".bin";
    ofstream outFile(fileName, ios::out | ios::binary);
    if(!outFile.is_open())return;
    outFile.write((char*)(&m),sizeof(int ));
    outFile.write((char*)(&nnz),sizeof(int ));
    outFile.write((char*)(part),sizeof(int )*m);
    outFile.close();
    if(MtxToken== nullptr)return;
}
typedef pair<int,int> metis_pd;
template<typename V>
void metis_partitioning_inner(
        int m, int nnz,
        int nParts,
        int *RowPtr,
        int *ColIdx,
        int *part,
        V *val,const char *MtxToken) {
    int nWeights = 1;
    int objval;
    int *cpyRowPtr = (int *) malloc(sizeof(int) * (m + 1));

    int *cpyColIdx = (int *) malloc(sizeof(int) * nnz);
    int *ColIdxOrder = (int *) malloc(sizeof(int) * nnz);

    int *revIndex = (int *) malloc(sizeof(int) * m);

    V *cpyVal = (V*)malloc(sizeof(V) * nnz);

    memcpy(cpyVal, val, sizeof(V)  * nnz);

    memcpy(cpyRowPtr, RowPtr, sizeof(int) * (m + 1));

    memcpy(cpyColIdx, ColIdx, sizeof(int) * nnz);
    timeval t1,t2;
    if(MtxToken== nullptr
    || !findToken(m,nnz,MtxToken,part)
    ) {
        int ret = METIS_PartGraphKway(&m, &nWeights, cpyRowPtr, cpyColIdx,
                                      NULL, NULL, NULL, &nParts, NULL,
                                      NULL, NULL, &objval, part);
        loadToken(m,nnz,MtxToken,part);
    }

    auto order = (metis_pd*)malloc(sizeof(metis_pd) * max(m,nnz));
    for (int i = 0; i < m; ++i) {
        order[i] = make_pair(part[i],i);
    }
    sort(order,order+m);
    for (int i = 0; i < m; ++i) {
        part[i] = order[i].second;
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
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        int len = cpyRowPtr[i + 1] - cpyRowPtr[i];

        for (int j = RowPtr[part[i]]; j < RowPtr[part[i] + 1]; ++j) {

            order[j ].first = part[ColIdx[j]];
            order[j ].second = j;
        }
        //qsort(order, len, sizeof(Pair_t), cmp_pair);
        sort(order+RowPtr[part[i]],order+RowPtr[part[i] + 1]);
        for (int j = 0; j < len; ++j) {

            cpyColIdx[j + cpyRowPtr[i]] = ColIdx[order[j + RowPtr[part[i]]].second];
            cpyVal[j + cpyRowPtr[i]] = val[order[j + RowPtr[part[i]]].second];
        }
    }


    memcpy(val, cpyVal, sizeof(V) * nnz);

    memcpy(RowPtr, cpyRowPtr, sizeof(int) * (m + 1));

    memcpy(ColIdx, cpyColIdx, sizeof(int) * (nnz));
    for(int j = RowPtr[0] ; j < RowPtr[1] ; ++j){
        //printf("%d %d\n",part[ColIdx[j]],ColIdx[j]);
    }
    free(order);
    free(cpyVal);
    free(cpyColIdx);
    free(cpyRowPtr);
    free(revIndex);
}
void metis_partitioning(
        int m, int nnz,
        int nParts,
        int *RowPtr,
        int *ColIdx,
        int *part,
        void *val, BASIC_SIZE_TYPE size,const char *MtxToken) {
    if(size==sizeof(double )){
        metis_partitioning_inner(m,nnz,nParts,RowPtr,ColIdx,part,(double*)val,MtxToken);
    }else{
        metis_partitioning_inner(m,nnz,nParts,RowPtr,ColIdx,part,(float*)val,MtxToken);
    }
}

template<typename V>
void ReGather_inner(V *true_val, const V *val, const int *index, int len) {
    //memcpy(true_val, val, size * len);
    for (int i = 0; i < len; ++i) {
        true_val[index[i]] = val[i];
    }
}
void ReGather(void *true_val, const void *val, const int *index, BASIC_SIZE_TYPE size, int len) {
    //memcpy(true_val, val, size * len);
    if (size == sizeof(double)) {
        ReGather_inner((double *)true_val,(double *)val,index,len);
    } else {
        ReGather_inner((float *)true_val,(float *)val,index,len);
    }
}
#endif