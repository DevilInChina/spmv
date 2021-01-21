#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <sys/time.h>
#include "sys/sysinfo.h"
#include "mmio_highlevel.h"
#include <metis.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE float
#endif
#ifndef PARTS
#define PARTS 8
#endif
#ifndef NTIMES
#define NTIMES 200
#endif

void writeresults(char *filename_res, char *filename, int m, int n, int nnzR, double time, double GFlops, int nParts,
                  int nthreads, double bandwidth) {
    FILE *fres = fopen(filename_res, "a");
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s,%i,%i,%d,%lf,%lf,%i,%i,%lf\n", filename, m, n, nnzR, time, GFlops, nParts, nthreads, bandwidth);
    fclose(fres);
}

int **subrowptrA, **subcolidxA;
VALUE_TYPE **subvalA, **X, **Y;

typedef struct numaspmv {
    int nthreads, numanodes, m, coreidx, alloc, *subX_ex, *subX, *rowPtr_RS, *colIdx_RS_CS;
    VALUE_TYPE *value, **Y, **X;
} numaspmv;

typedef struct thread {
    int nthreads, m, coreidx, threadidx, *rowptr, *colidx, *rowPtr_RS, *colIdx_RS_CS;
    VALUE_TYPE *value, *Y, *X;
} thread;

void metis_partitioning(int n, int m, int nnz, idx_t nParts, int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *val) {
    //idx_t nn=n;
    //idx_t nParts=2;
    idx_t nn = n;
    //idx_t nParts=2;
    idx_t nWeights = 1;
    idx_t *part = (idx_t *) malloc(sizeof(idx_t) * (nn + 1));
    idx_t objval;
    idx_t *csrRowPtrAAA = (idx_t *) malloc(sizeof(idx_t) * (nn + 1));
    idx_t *csrColIdxAAA = (idx_t *) malloc(sizeof(idx_t) * nnz);
    int i, j, p, q;
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
    exclusive_scan(rowPtr_RS, n + 1);
    //colnum sorting
    //csr to csc
    int *rowIdx_RS = (int *) malloc(sizeof(int) * nnz);
    int *colPtr_RS = (int *) malloc(sizeof(int) * (n + 1));
    VALUE_TYPE *val_RS = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * nnz);
    matrix_transposition(n, n, nnz, rowPtr_RS, colIdx_RS, val,
                         rowIdx_RS, colPtr_RS, val_RS);
    colPtr_RS[n] = nnz;
    //int * rowPtr_RS_CS = (int *)malloc(sizeof(int) * (n+1));
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

}

void *spmv(void *arg) {
    numaspmv *pn = (numaspmv *) arg;
    int me = pn->alloc;
    numa_run_on_node(me);
    int m = pn->m;
    int nthreads = pn->nthreads;
    int numanodes = pn->numanodes;
    int coreidx = pn->coreidx;
    int eachnumathreads = nthreads / numanodes;
    int task = ceil((double) m / (double) eachnumathreads);
    int start = coreidx * task;
    int end = (coreidx + 1) * task > m ? m : (coreidx + 1) * task;
    //printf("numanode %d, coreindex %d, m %d, nthreads %d, eachnumathreads %d, start %d, end %d\n",pn->alloc,coreidx,pn->m,nthreads,eachnumathreads,start,end);
    VALUE_TYPE *val = subvalA[me];
    //VALUE_TYPE *x = X[me];
    VALUE_TYPE *y = Y[me];
    int *rpt = subrowptrA[me];
    int *col = subcolidxA[me];
    for (int u = start; u < end; u++) {
        VALUE_TYPE sum = 0;
        for (int j = rpt[u]; j < rpt[u + 1]; j++) {
            int Xpos = col[j] / pn->subX[0];
            int remainder = col[j] - pn->subX_ex[Xpos];
            sum += val[j] * X[Xpos][remainder];
        }
        y[u] = sum;
        //if(me==7)
        //printf("y[%d][%d]%.2f\n",me,u,sum);
    }
}

void *pthreadspmv(void *arg) {

    thread *pn = (thread *) arg;
    int me = pn->coreidx;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(me, &cpuset);
    //bind to me
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    int m = pn->m;
    int nthreads = pn->nthreads;
    int threadidx = pn->threadidx;
    int task = ceil((double) m / (double) nthreads);
    int start = threadidx * task;
    int end = (threadidx + 1) * task > m ? m : (threadidx + 1) * task;
    for (int i = start; i < end; i++) {
        VALUE_TYPE sum = 0;
        for (int j = pn->rowptr[i]; j < pn->rowptr[i + 1]; j++) {
            sum += pn->value[j] * pn->X[pn->colidx[j]];
        }
        pn->Y[i] = sum;
    }

}

int main(int argc, char **argv) {

    char *filename = argv[1];
    int i, j, k;
    int n, m, nnz, isSymmetricr;
    int *csrRowPtrA, *csrColIdxA;
    VALUE_TYPE *csrValA;
    int *rowPtr_RS;
    int *colIdx_RS_CS;

    idx_t nParts = atoi(argv[2]);
    printf("filename = %s\n", filename);
    mmio_allinone(&m, &n, &nnz, &isSymmetricr, &csrRowPtrA, &csrColIdxA, &csrValA, filename);

    printf("Matrix A is %i by %i, #nonzeros = %i\n", m, n, nnz);
    metis_partitioning(n, m, nnz, nParts, csrRowPtrA, csrColIdxA, csrValA);

    struct timeval t1, t2;
    int cores = get_nprocs_conf();
    int numanodes = numa_max_node() + 1;
    int nthreads = atoi(argv[3]);
    int eachnumacores = nthreads / numanodes;

    if (numanodes <= 1) {
        printf("There are no NUMA nodes, it will bind threads to cores!\n");
        VALUE_TYPE *vector = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * n);
        VALUE_TYPE *Y = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * m);
        memset(vector, 0, sizeof(VALUE_TYPE) * n);
        memset(Y, 0, sizeof(VALUE_TYPE) * m);
        for (i = 0; i < n; i++) {
            vector[i] = rand() % 10;
        }

        pthread_t *threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
        pthread_attr_t pthread_custom_attr;
        pthread_attr_init(&pthread_custom_attr);

        thread *p = (thread *) malloc(nthreads * sizeof(thread));
        for (i = 0; i < nthreads; i++) {
            p[i].threadidx = i;
            p[i].coreidx = i % cores;
            p[i].nthreads = nthreads;
            p[i].value = csrValA;
            p[i].m = m;
            p[i].X = vector;
            p[i].Y = Y;
            p[i].rowptr = csrRowPtrA;
            p[i].colidx = csrColIdxA;
        }
        gettimeofday(&t1, NULL);
        for (int r = 0; r < NTIMES; r++) {
            for (i = 0; i < nthreads; i++) {
                pthread_create(&threads[i], &pthread_custom_attr, pthreadspmv, (void *) (p + i));
            }
            for (i = 0; i < nthreads; i++) {
                pthread_join(threads[i], NULL);
            }
        }
        gettimeofday(&t2, NULL);

        double time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / NTIMES;
        double GFlops = 2 * nnz / time / pow(10, 6);
        double bandwidth =
                (((m + 1) + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(VALUE_TYPE)) * nthreads / time / pow(10, 6);
        printf("bind thread spmv time %.2f  GFlops %.2f  bandwidth %.2f \n", time, GFlops, bandwidth);

        VALUE_TYPE *Y_golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * m);
        memset(Y_golden, 0, sizeof(VALUE_TYPE) * m);
        for (i = 0; i < m; i++) {
            Y_golden[i] = 0;
            for (j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++) {
                Y_golden[i] += csrValA[j] * vector[csrColIdxA[j]];
            }

        }
        int errorcount = 0;
        for (i = 0; i < m; i++) {
            if (Y_golden[i] != Y[i]) {
                errorcount++;
            }

        }
        printf("error count %d\n", errorcount);
    } else {
        printf("There are %d numa nodes, %d cores, each numa node has %d cores\n", numanodes, cores, eachnumacores);

        VALUE_TYPE *vector = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (n + 1));
        memset(vector, 0, sizeof(VALUE_TYPE) * (n + 1));
        for (i = 0; i < n; i++)
            vector[i] = rand()%8*0.125;

        //partition
        int *subrowpos = (int *) malloc(sizeof(int) * (PARTS + 1));
        int *subm = (int *) malloc(sizeof(int) * PARTS);
        int *subm_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
        int *subX = (int *) malloc(sizeof(int) * PARTS);
        int *subX_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
        int *subnnz = (int *) malloc(sizeof(int) * PARTS);
        int *subnnz_ex = (int *) malloc(sizeof(int) * (PARTS + 1));
        for (i = 0; i < PARTS; i++)
            subrowpos[i] = (ceil((double) m / (double) PARTS)) * i > m ? m : ((ceil((double) m / (double) PARTS)) * i);
        subrowpos[PARTS] = m;
        for (i = 0; i < (PARTS - 1); i++) {
            subX[i] = ceil((double) n / (double) PARTS);
        }
        subX[PARTS - 1] = n - subX[0] * (PARTS - 1);
        for (i = 0; i < PARTS; i++) {
            subX_ex[i] = subX[i];
        }
        exclusive_scan(subX_ex, PARTS + 1);
        for (i = 0; i < PARTS; i++) {
            subm[i] = subrowpos[i + 1] - subrowpos[i];
            subm_ex[i] = subrowpos[i + 1] - subrowpos[i];
        }
        exclusive_scan(subm_ex, PARTS + 1);
        for (i = 0; i < PARTS; i++) {
            subnnz[i] = csrRowPtrA[subrowpos[i] + subm[i]] - csrRowPtrA[subrowpos[i]];
            subnnz_ex[i] = subnnz[i];
        }
        exclusive_scan(subnnz_ex, PARTS + 1);
        numaspmv *p = (numaspmv *) malloc(nthreads * sizeof(numaspmv));
        pthread_t *threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
        pthread_attr_t pthread_custom_attr;
        pthread_attr_init(&pthread_custom_attr);
        for (i = 0; i < nthreads; i++) {
            p[i].alloc = i % numanodes;
            p[i].numanodes = numanodes;
            p[i].nthreads = nthreads;
            p[i].subX = subX;
            p[i].subX_ex = subX_ex;
        }
        for (i = 0; i < eachnumacores; i++) {
            for (j = 0; j < numanodes; j++) {
                p[i * numanodes + j].coreidx = i;
                p[i * numanodes + j].m = subm[j];
            }
        }
        subrowptrA = (int **) malloc(sizeof(int *) * nthreads);
        subcolidxA = (int **) malloc(sizeof(int *) * nthreads);
        subvalA = (VALUE_TYPE **) malloc(sizeof(VALUE_TYPE *) * nthreads);
        X = (VALUE_TYPE **) malloc(sizeof(VALUE_TYPE *) * nthreads);
        Y = (VALUE_TYPE **) malloc(sizeof(VALUE_TYPE *) * nthreads);

        for (i = 0; i < nthreads; i++) {
            subrowptrA[i] = numa_alloc_onnode(sizeof(int) * (subm[p[i].alloc] + 1), p[i].alloc);
            subcolidxA[i] = numa_alloc_onnode(sizeof(int) * subnnz[p[i].alloc], p[i].alloc);
            subvalA[i] = numa_alloc_onnode(sizeof(VALUE_TYPE) * subnnz[p[i].alloc], p[i].alloc);
            X[i] = numa_alloc_onnode(sizeof(VALUE_TYPE) * subX[p[i].alloc], p[i].alloc);
            Y[i] = numa_alloc_onnode(sizeof(VALUE_TYPE) * subm[p[i].alloc], p[i].alloc);
        }
        /*for(i = 0; i < numanodes; i++)
        {
            for(j = 0; j < eachnumacores; j++)
            {
                for (k = 0; k <= subm[i]; k++)
                {
                    subrowptrA[i+j*eachnumacores][k] = rowptrA[subrowpos[i]+k];
                }
            }
            for(j = 0; j < eachnumacores; j++)
            {
                for (k = 0; k < subnnz[i]; k++)
                {
                    subcolidxA[i+j*eachnumacores][k] = colidxA[subnnz_ex[i]+k];
                    subvalA[i+j*eachnumacores][k] = valA[subnnz_ex[i]+k];
                }
            }
            for(j = 0; j < eachnumacores; j++)
            {
                for (k = 0; k < subX[i]; k++)
                {
                    X[i+j*eachnumacores][k] = vector[subX_ex[i]+k];
                }
            }

        }*/
        int currentcore = 0;
        for (i = 0; i < numanodes; i++) {
            for (j = 0; j < eachnumacores; j++) {
                for (k = 0; k <= subm[i]; k++) {
                    currentcore = i + j * eachnumacores;
                    if (currentcore < nthreads) {
                        subrowptrA[currentcore][k] = csrRowPtrA[subrowpos[i] + k];
                    }
                }
            }
            for (j = 0; j < eachnumacores; j++) {
                for (k = 0; k < subnnz[i]; k++) {
                    currentcore = i + j * eachnumacores;
                    if (currentcore < nthreads) {
                        subcolidxA[currentcore][k] = csrColIdxA[subnnz_ex[i] + k];
                        subvalA[currentcore][k] = csrValA[subnnz_ex[i] + k];
                    }
                }
            }
            for (j = 0; j < eachnumacores; j++) {
                for (k = 0; k < subX[i]; k++) {
                    currentcore = i + j * eachnumacores;
                    if (currentcore < nthreads) {
                        X[currentcore][k] = vector[subX_ex[i] + k];
                    }
                }
            }

        }
        for (i = 0; i < nthreads; i++) {
            if (i % numanodes != 0) {
                int temprpt = subrowptrA[i][0];
                for (j = 0; j <= subm[i % numanodes]; j++) {
                    subrowptrA[i][j] -= temprpt;
                }
            }
        }

        gettimeofday(&t1, NULL);
        for (int r = 0; r < NTIMES; r++) {
            for (i = 0; i < nthreads; i++) {
                pthread_create(&threads[i], &pthread_custom_attr, spmv, (void *) (p + i));
            }
            for (i = 0; i < nthreads; i++) {
                pthread_join(threads[i], NULL);
            }
        }
        gettimeofday(&t2, NULL);
        double time_numa = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / NTIMES;
        double GFlops_numaspmv = 2 * nnz / time_numa / pow(10, 6);
        double bandwidth = (((m + 1) + nnz) * sizeof(int) + (2 * nnz + m) * sizeof(VALUE_TYPE)) * nthreads / time_numa /
                           pow(10, 6);
        printf("numaspmv time %.2f  GFlops_numaspmv %.2f  bandwidth %.2f \n", time_numa, GFlops_numaspmv, bandwidth);

        VALUE_TYPE *Y_gather = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * m);
        memset(Y_gather, 0, sizeof(VALUE_TYPE) * m);
        VALUE_TYPE *Y_golden = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * m);
        memset(Y_golden, 0, sizeof(VALUE_TYPE) * m);
        for (i = 0; i < PARTS; i++) {
            for (j = 0; j < subm[i]; j++) {
                Y_gather[subm_ex[i] + j] = Y[i][j];
            }

        }
        for (i = 0; i < m; i++) {
            Y_golden[i] = 0;
            for (j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++) {
                Y_golden[i] += csrValA[j] * vector[csrColIdxA[j]];
            }

        }
        int errorcount = 0;
        for (i = 0; i < m; i++) {
            if (Y_golden[i] != Y_gather[i]) {
                errorcount++;
            }

        }
        printf("error count %d\n", errorcount);


        //writeresults("spmv_numa.csv", filename,m,n,nnz,time_numa,GFlops_numaspmv,nthreads,bandwidth);
    }
}