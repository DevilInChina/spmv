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

#ifndef PARTS
#define PARTS 2
#endif
#ifndef NTIMES
#define NTIMES 1
#endif

void writeresults(char *filename_res, char *filename, int m, int n, int nnzR, double time, double GFlops, int nthreads,
                  double bandwidth) {
    FILE *fres = fopen(filename_res, "a");
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s,%i,%i,%d,%lf,%lf,%i,%lf\n", filename, m, n, nnzR, time, GFlops, nthreads, bandwidth);
    fclose(fres);
}

int **subrowptrA, **subcolidxA;
VALUE_TYPE **subvalA, **X, **Y;

typedef struct numaspmv {
    int nthreads, numanodes, m, coreidx, alloc, *subX_ex, *subX;
    VALUE_TYPE *value, **Y, **X;
} numaspmv;

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

int main(int argc, char **argv) {
    int m, n, nnzA, isSymmetric;
    int i, j, k;
    int *rowptrA;
    int *colidxA;
    VALUE_TYPE *valA;
    struct timeval t1, t2;
    int cores = get_nprocs_conf();
    int numanodes = numa_max_node() + 1;
    int nthreads = atoi(argv[2]);
    int eachnumacores = nthreads / numanodes;

    if (numa_available() < 0) {
        printf("Your system does not support NUMA API\n");
        return 0;
    }
    printf("There are %d numa nodes, %d cores, each numa node has %d cores\n", numanodes, cores, eachnumacores);
    char *filename = argv[1];
    printf("filename = %s\n", filename);
    mmio_allinone(&m, &n, &nnzA, &isSymmetric, &rowptrA, &colidxA, &valA, filename);
    printf("Matrix A is %i by %i, #nonzeros = %i\n", m, n, nnzA);

    for (i = 0; i < nnzA; i++) {
        valA[i] = 1.0;
    }
    VALUE_TYPE *vector = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * n);
    memset(vector, 0, sizeof(VALUE_TYPE) * n);
    for (i = 0; i < n; i++) {
        vector[i] = rand() % 10;
        //printf("vector[%d] %f\n", i ,vector[i]);
    }

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
        subnnz[i] = rowptrA[subrowpos[i] + subm[i]] - rowptrA[subrowpos[i]];
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
                    subrowptrA[currentcore][k] = rowptrA[subrowpos[i] + k];
                }
            }
        }
        for (j = 0; j < eachnumacores; j++) {
            for (k = 0; k < subnnz[i]; k++) {
                currentcore = i + j * eachnumacores;
                if (currentcore < nthreads) {
                    subcolidxA[currentcore][k] = colidxA[subnnz_ex[i] + k];
                    subvalA[currentcore][k] = valA[subnnz_ex[i] + k];
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
    double GFlops_numaspmv = 2 * nnzA / time_numa / pow(10, 6);
    double bandwidth =
            (((m + 1) + nnzA) * sizeof(int) + (2 * nnzA + m) * sizeof(VALUE_TYPE)) * nthreads / time_numa / pow(10, 6);
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
        for (j = rowptrA[i]; j < rowptrA[i + 1]; j++) {
            Y_golden[i] += valA[j] * vector[colidxA[j]];
        }

    }
    int errorcount = 0;
    for (i = 0; i < m; i++) {
        if (Y_golden[i] != Y_gather[i]) {
            errorcount++;
        }

    }
    printf("error count %d\n", errorcount);

    //writeresults("spmv_numa.csv", filename,m,n,nnzA,time_numa,GFlops_numaspmv,nthreads,bandwidth);
    /*double row_length_mean = ((double)nnzA)/m;
    double variance = 0.0;
    double row_length_skewness = 0.0;

    for(int row = 0; row < m; ++row)
    {
            int length = rowptrA[row + 1] - rowptrA[row];
            double delta = (double)(length - row_length_mean);
            variance += (delta *delta);
            row_length_skewness += (delta *delta *delta);

    }
    variance /= m;
    double row_length_std_dev = sqrt(variance);
    row_length_skewness = (row_length_skewness / m) / pow(row_length_std_dev,3.0);
    double row_length_variation = row_length_std_dev / row_length_mean;
    printf("%lf\n",row_length_variation);
*/

}
