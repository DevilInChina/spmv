#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h> 
#include "data-types.h"
#include "mmio_highlevel.h"
#include <spmv.h>
#define min(a,b) ((a<b)?(a):(b)) 
/* run this program using the console pauser or add your own getch, system("pause") or input loop */

/*-------------------------------------------------*/
/* sets parameters required by arms preconditioner */
/* input ITS_IOT, Dscale                            */
/* output ipar tolcoef, lfil                       */
/*-------------------- trigger an error if not set */
void * itsol_malloc(int nbytes, char *msg)//malloc 操作
{
     void *ptr;

     if (nbytes == 0) return NULL;

     ptr = (void *)malloc(nbytes);
    if (ptr == NULL)
         //itsol_errexit("Not enough mem for %s. Requested size: %d bytes", msg, nbytes);

     return ptr;
}
double itsol_norm(double *x, int n)
{
    int i;
    double t = 0.;
    
    assert(n >= 0);
    if (n > 0) assert(x != NULL);

    for (i = 0; i < n; i++)  t += x[i] * x[i];

    return sqrt(t);
}
double itsol_dot(double *x, double *y, int n)
{
    int i;
    double t = 0.;
    
    assert(n >= 0);
    if (n > 0) assert(x != NULL && y != NULL);

    for (i = 0; i < n; i++)  t += x[i] * y[i];

    return t;
}
int itsol_setupCS(ITS_SparMat *amat, int len, int job)//初始化矩阵
{
    amat->n = len;
    amat->nzcount = (int *)itsol_malloc(len * sizeof(int), "itsol_setupCS");
    amat->ja = (int **)itsol_malloc(len * sizeof(int *), "itsol_setupCS");
    if (job == 1)
        amat->ma = (double **)itsol_malloc(len * sizeof(double *), "itsol_setupCS");
    else
        amat->ma = NULL;
    return 0;
}
ITS_CooMat itsol_read_coo(char *Fname)
{
    FILE *matf = NULL;
    double *aa;
    int *ii, *jj;
    int k, n, nnz;
    ITS_CooMat A;
    
    char str[ITS_MAX_LINE];

    /*-------------------- start */
    if ((matf = fopen(Fname, "r")) == NULL) {
        fprintf(stdout, "Cannot Open Matrix\n");
        exit(3);
    }
    
    /*-------------------- mtx format .. in some cases n, 
      nnz are read separately per line */
    /*-------------------- try a 100 lines max of comments */
    for (k = 0; k < 100; k++) {
        fgets(str, ITS_MAX_LINE, matf);

        if (memcmp(str, "%", sizeof(char)) != 0) break;
    }
    
    if (k == 99) exit(3);

    sscanf(str, " %d %d %d", &n, &k, &nnz);
    if (n != k) {
        fprintf(stdout, "This is not a square matrix -- stopping \n");
        exit(4);
    }

    /* separate reads for n and nnz 
       fscanf(matf," %d", &n); 
       fscanf(matf," %d", &nnz);  
     */
    //bzero(&A, sizeof(A));
    memset(&A,0,sizeof(A));
    A.n = n;
    A.nnz = nnz;

    aa = A.ma = (double *)itsol_malloc(nnz * sizeof(double), "read_coo:3");
    jj = A.ja = (int *)itsol_malloc(nnz * sizeof(int), "read_coo:4");
    ii = A.ia = (int *)itsol_malloc(nnz * sizeof(int), "read_coo:5");
    /*-------------------- long live fortran77 --- */
    for (k = 0; k < nnz; k++) {
        fscanf(matf, "%d  %d  %s", &ii[k], &jj[k], str);
        aa[k] = atof(str);
    }

    fclose(matf);

    return A;
}
void itsol_pc_initialize(ITS_PC *pc, ITS_PC_TYPE pctype)
{
    assert(pc != NULL);
    pc->pc_type = pctype;

    if (pctype == ITS_PC_ILUC || pctype == ITS_PC_ILUK || pctype == ITS_PC_ILUT) {
        pc->ILU = (ITS_ILUSpar *) itsol_malloc(sizeof(ITS_ILUSpar), "pc init");
    }
    else {
        fprintf(pc->log, "wrong preconditioner type\n");
        exit(-1);
    }
}
void itsol_set_arms_pars(ITS_PARS *io, int Dscale, int *ipar, double *dropcoef, int *lfil)
{
    int j;

    for (j = 0; j < 17; j++)
        ipar[j] = 0;

    /*-------------------- */
    ipar[0] = 10;      /* max number of levels allowed */
    ipar[1] = io->perm_type;    /* Indset (0) / PQ (1)    permutation   */

    /* note that these refer to completely  */
    /* different methods for reordering A   */
    /* 0 = standard ARMS independent sets   */
    /* 1 = arms with ddPQ ordering          */
    /* 2 = acoarsening-based ordering [new] */

    ipar[2] = io->Bsize;        /* smallest size allowed for last schur comp. */
    ipar[3] = 1;                /* whether or not to print statistics */

    /*-------------------- interlevel methods */
    ipar[10] = 0;               /* Always do permutations - currently not used  */
    ipar[11] = 0;               /* ILUT or ILUTP - currently only ILUT is implemented */
    ipar[12] = Dscale;          /* diagonal row scaling before PILUT 0:no 1:yes */
    ipar[13] = Dscale;          /* diagonal column scaling before PILUT 0:no 1:yes */

    /*-------------------- last level methods */
    ipar[14] = 1;               /* Always do permutations at last level */
    ipar[15] = 1;               /* ILUTP for last level(0 = ILUT at last level) */
    ipar[16] = Dscale;          /* diagonal row scaling  0:no 1:yes */
    ipar[17] = Dscale;          /* diagonal column scaling  0:no 1:yes */

    /*-------------------- set lfil */
    for (j = 0; j < 7; j++) {
        lfil[j] = io->ilut_p;
    }

    /*--------- dropcoef (droptol[k] = tol0*dropcoef[k]) ----- */
    dropcoef[0] = 1.6;          /* dropcoef for L of B block */
    dropcoef[1] = 1.6;          /* dropcoef for U of B block */
    dropcoef[2] = 1.6;          /* dropcoef for L\ inv F */
    dropcoef[3] = 1.6;          /* dropcoef for U\ inv E */
    dropcoef[4] = 0.004;        /* dropcoef for forming schur comple. */
    dropcoef[5] = 0.004;        /* dropcoef for last level L */
    dropcoef[6] = 0.004;        /* dropcoef for last level U */
}
void itsol_solver_init_pars(ITS_PARS *p)
{
    assert(p != NULL);

    p->fp = stdout;
    p->verb = 2;

    /* parameters from inputs -----------------------------------------*/
    p->bgsl = 4;
    p->restart = 30;               /* Dim of Krylov subspace [fgmr]   */
    p->maxits = 1000;              /* maximum number of fgmres iters  */
    p->tol = 1e-5;                 /* tolerance for stopping fgmres   */

    p->eps = 0.8;
    p->ilut_p = 50;                /* initial lfil                    */
    p->ilut_tol = 1e-3;            /* initial drop tolerance          */
    p->iluk_level = 1;             /* initial level of fill for ILUK  */

    /* value always set to 1           */
    p->perm_type = 0;              /* indset perms (0) or PQ perms (1)*/
    p->Bsize = 30;                 /* block size - dual role. see input file */

    /* arms */
    p->diagscal = 1;
    p->tolind = ITS_TOL_DD;

    /* init arms pars */
    itsol_set_arms_pars(p, p->diagscal, p->ipar, p->dropcoef, p->lfil_arr);
}
void itsol_solver_initialize(ITS_SOLVER *s, ITS_SOLVER_TYPE stype, ITS_PC_TYPE pctype)
{
    assert(s != NULL);
    /* init */
    //bzero(s, sizeof(*s));
    memset(s,0,sizeof(*s));
    s->s_type = stype;
    s->log = stdout;
    
    /* pc */
    s->pc_type = pctype;
    s->pc.log = s->log;
    itsol_pc_initialize(&s->pc, pctype);
    /* init parameters */
    itsol_solver_init_pars(&s->pars);
}
int itsol_setupILU(ITS_ILUSpar *lu, int n,int nnzr)
{
    lu->n = n;
    lu->D = (double *)itsol_malloc(sizeof(double) * n, "itsol_setupILU");
    lu->L = (ITS_SparMat *) itsol_malloc(sizeof(ITS_SparMat), "itsol_setupILU");

    lu->L->RowPtr=(int *)malloc(n*sizeof(int));
    lu->L->ColIdx=(int *)malloc(nnzr*sizeof(int));
    lu->L->Val=(double *)malloc(nnzr*sizeof(double));
    lu->L->RowPtr[0]=0;

    lu->U = (ITS_SparMat *) itsol_malloc(sizeof(ITS_SparMat), "itsol_setupILU");

    lu->U->RowPtr=(int *)malloc(n*sizeof(int));
    lu->U->ColIdx=(int *)malloc(nnzr*sizeof(int));
    lu->U->Val=(double *)malloc(nnzr*sizeof(double));
    lu->U->RowPtr[0]=0;
    lu->work = (int *)itsol_malloc(sizeof(int) * n, "itsol_setupILU");

    return 0;
}
/*--------------------------------------------------------------------
 * symbolic ilu factorization to calculate structure of ilu matrix
 * for specified level of fill
 *--------------------------------------------------------------------
 * on entry:
 * =========
 * lofM     = level of fill, lofM >= 0
 * csmat    = matrix stored in SpaFmt format -- see heads.h for details
 *            on format
 * lu       = pointer to a ILUSpar struct -- see heads.h for details
 *            on format
 * fp       = file pointer for error log ( might be stderr )
 *--------------------------------------------------------------------
 * on return:
 * ==========
 * ierr     = return value.
 *            ierr  = 0   --> successful return.
 *            ierr != 0   --> error
 * lu->n    = dimension of the block matrix
 *   ->L    = L part -- stored in SpaFmt format, patterns only in lofC
 *   ->U    = U part -- stored in SpaFmt format, patterns only in lofC
 *------------------------------------------------------------------*/
int itsol_pc_lofC(int lofM, ITS_SparMat *csmat, ITS_ILUSpar *lu, FILE * fp)
{
    //csmat和L以及U是二维的
    int n = csmat->n;
    int *levls = NULL, *jbuf = NULL, *iw = lu->work;
    int **ulvl;                 /*  stores lev-fils for U part of ILU factorization */
    ITS_SparMat *L = lu->L, *U = lu->U;
   //将二维转换成一维
    //转换结束
    /*--------------------------------------------------------------------
     * n        = number of rows or columns in matrix
     * inc      = integer, count of nonzero(fillin) element of each row
     *            after symbolic factorization
     * ju       = entry of U part of each row
     * lvl      = buffer to store levels of each row
     * jbuf     = buffer to store column index of each row
     * iw       = work array
     *------------------------------------------------------------------*/
    int i, j, k, col, ip, it, jpiv;
    int incl, incu, jmin, kmin;

    (void)fp;

    levls = (int *)itsol_malloc(n * sizeof(int), "lofC");
    jbuf = (int *)itsol_malloc(n * sizeof(int), "lofC");
    ulvl = (int **)itsol_malloc(n * sizeof(int *), "lofC");

    /* initilize iw */
    for (j = 0; j < n; j++)
        iw[j] = -1;
    int sum=0;
    int sum_l=0;
    for(i=0;i<n;i++)
    {
        incl=0;
        incu=i;
        /*-------------------- assign lof = 0 for matrix elements */
        for(int j=csmat->RowPtr[i];j<csmat->RowPtr[i+1];j++)
        {
            col=csmat->ColIdx[j];
            if (col < i) {//initiliza the L part
                /*-------------------- L-part  */
                jbuf[incl] = col;//store the col index
                levls[incl] = 0;//store the level
                iw[col] = incl++;//store the index
            }
            else if (col > i) {//
                /*-------------------- U-part  */
                jbuf[incu] = col;//
                levls[incu] = 0;
                iw[col] = incu++;//
            }
        }
        /*-------------------- symbolic k,i,j Gaussian elimination  */
        jpiv = -1;
        while (++jpiv < incl) {
            k = jbuf[jpiv];
            /*-------------------- select leftmost pivot */
            kmin = k;
            jmin = jpiv;
            for (j = jpiv + 1; j < incl; j++) {
                if (jbuf[j] < kmin) {
                    kmin = jbuf[j];
                    jmin = j;
                }
            }
            /*-------------------- swap  */
            if (jmin != jpiv) {
                jbuf[jpiv] = kmin;
                jbuf[jmin] = k;
                iw[kmin] = jpiv;
                iw[k] = jmin;
                j = levls[jpiv];
                levls[jpiv] = levls[jmin];
                levls[jmin] = j;
                k = kmin;
            }
        /*-------------------- symbolic linear combinaiton of rows  */
            for(j=U->RowPtr[k];j<U->RowPtr[k+1];j++)
            {
                col=U->ColIdx[j];
                it = ulvl[k][j-U->RowPtr[k]] + levls[jpiv] + 1;
                if (it > lofM)
                    continue;
                ip = iw[col];
                if (ip == -1) {
                    if (col < i) {
                        jbuf[incl] = col;
                        levls[incl] = it;
                        iw[col] = incl++;
                    }
                    else if (col > i) {
                        jbuf[incu] = col;
                        levls[incu] = it;
                        iw[col] = incu++;
                    }
                }
                else
                    levls[ip] = min(levls[ip], it);
            }
        }
        /*-------------------- reset iw */
         for (j = 0; j < incl; j++)
            iw[jbuf[j]] = -1;
        for (j = i; j < incu; j++)
            iw[jbuf[j]] = -1;
        /*-------------------- copy L-part */
        sum_l+=incl;
        L->RowPtr[i+1]=sum_l;
        if (incl > 0) {
            for(int w=0;w<incl;w++)
            {
                L->ColIdx[L->RowPtr[i]+w]=jbuf[w];
            }
        }
        /*-------------------- copy U - part        */
        k = incu - i;
        sum+=k;
        U->RowPtr[i+1]=sum;
        if (k > 0) {
            for(int w=0;w<k;w++)
            {
                U->ColIdx[U->RowPtr[i]+w]=jbuf[i+w];
            }
            /*-------------------- update matrix of levels */
            ulvl[i] = (int *)itsol_malloc(k * sizeof(int), "lofC");
            memcpy(ulvl[i], levls + i, k * sizeof(int));    

    }
    }
    /*-------------------- free temp space and leave --*/
//    free(levls);
//    free(jbuf);
//    free(ulvl);

    return 0;
}
/*----------------------------------------------------------------------------
 * ILUK preconditioner
 * incomplete LU factorization with level of fill dropping
 *----------------------------------------------------------------------------
 * Parameters
 *----------------------------------------------------------------------------
 * on entry:
 * =========
 * lofM     = level of fill: all entries with level of fill > lofM are
 *            dropped. Setting lofM = 0 gives BILU(0).
 * csmat    = matrix stored in SpaFmt format -- see heads.h for details
 *            on format
 * lu       = pointer to a ILUKSpar struct -- see heads.h for details
 *            on format
 * fp       = file pointer for error log ( might be stderr )
 *
 * on return:
 * ==========
 * ierr     = return value.
 *            ierr  = 0   --> successful return.
 *            ierr  = -1  --> error in lofC
 *            ierr  = -2  --> zero diagonal found
 * lu->n    = dimension of the matrix
 *   ->L    = L part -- stored in SpaFmt format
 *   ->D    = Diagonals
 *   ->U    = U part -- stored in SpaFmt format
 *----------------------------------------------------------------------------
 * Notes:
 * ======
 * All the diagonals of the input matrix must not be zero
 *--------------------------------------------------------------------------*/
int itsol_pc_ilukC(int lofM, ITS_SparMat *csmat, ITS_ILUSpar *lu, FILE * fp)
{
    int ierr;
    int n = csmat->n;
    printf("n=%d\n",n);
    int *jw, i, j, k, col, jpos, jrow;
    ITS_SparMat *L, *U;
    double *D;
    int nnzr=csmat->RowPtr[n];
    printf("nnzr=%d\n",nnzr);
    itsol_setupILU(lu, n,nnzr);
    // L = lu->L;
    // U = lu->U;
    // D = lu->D;
    /* symbolic factorization to calculate level of fill index arrays */
    //level of fill
    printf("sxxx\n");
    if ((ierr = itsol_pc_lofC(lofM, csmat, lu, fp)) != 0) {
        fprintf(fp, "Error: lofC\n");
        return -1;
    }
    printf("sxxx\n");
    L = lu->L;
    U = lu->U;
    D = lu->D;
    jw = lu->work;//working buffer
    /* set indicator array jw to -1 */
    for (j = 0; j < n; j++)
        jw[j] = -1;

    /* beginning of main loop */
    for (i = 0; i < n; i++) {
        /* set up the i-th row accroding to the nonzero information from
           symbolic factorization */

        /* setup array jw[], and initial i-th row */
         /* initialize L part   */
         for(int j=L->RowPtr[i];j<L->RowPtr[i+1];j++)
        {
            col=L->ColIdx[j];
            jw[col]=j-L->RowPtr[i];
            L->Val[j]=0;
        }
        jw[i] = i;
        D[i] = 0;               /* initialize diagonal */
         /* initialize U part   */
         for(int j=U->RowPtr[i];j<U->RowPtr[i+1];j++)
        {
            col=U->ColIdx[j];
            jw[col]=j-U->RowPtr[i];
            U->Val[j]=0;
        }
        /* copy row from csmat into lu */
        for(j=csmat->RowPtr[i];j<csmat->RowPtr[i+1];j++)
        {
            col=csmat->ColIdx[j];
            jpos = jw[col];
            if (col < i)
            {
                L->Val[L->RowPtr[i]+jpos]=csmat->Val[j];
            }
            else if (col == i)
                D[i] = csmat->Val[j];
            else
            {
                U->Val[U->RowPtr[i]+jpos]=csmat->Val[j];

            }
        }
        for(j=L->RowPtr[i];j<L->RowPtr[i+1];j++)
        {
            jrow=L->ColIdx[j];
            L->Val[j]*=D[jrow];
            for(k=U->RowPtr[jrow];k<U->RowPtr[jrow+1];k++)
            {
                col=U->ColIdx[k];
                jpos=jw[col];
                if(jpos==-1)
                continue;
                if(col<i)
                {
                    L->Val[L->RowPtr[i]+jpos]-=L->Val[j]*U->Val[k];
                }
                else if(col == i)
                {
                    D[i]-=L->Val[j]*U->Val[k];
                }
                else
                {
                    U->Val[U->RowPtr[i]+jpos]-=L->Val[j]*U->Val[k];
                }
                
            }

        }

        /* reset double-pointer to -1 ( U-part) */
        for(j=L->RowPtr[i];j<L->RowPtr[i+1];j++)
        {
            col=L->ColIdx[j];
            jw[col]=-1;
        }
        jw[i] = -1;
        for(j=U->RowPtr[i];j<U->RowPtr[i+1];j++)
        {
            col=U->ColIdx[j];
            jw[col]=-1;
        }
        if (D[i] == 0) {
            fprintf(fp, "fatal error: Zero diagonal found...\n");
            return -2;
        }
        D[i] = 1.0 / D[i];
    }

    return 0;
}


int itsol_lusolC(double *y, double *x, ITS_ILUSpar *lu)//according y and lu solve x
{
    int n = lu->n, i, j, nzcount;
    double *D;
    ITS_SparMat *L, *U;

    L = lu->L;
    U = lu->U;
    D = lu->D;

    /* Block L solve */
    for (i = 0; i < n; i++) {
        x[i] = y[i];
        for(j=L->RowPtr[i];j<L->RowPtr[i+1];j++)
        {
            x[i]-=x[L->ColIdx[j]]*L->Val[j];
        }
    }
    /* Block -- U solve */
    for(i=n-1;i>=0;i--)
    {
        for(j=U->RowPtr[i];j<U->RowPtr[i+1];j++)
        {
            x[i]-=x[U->ColIdx[j]]*U->Val[j];
        }
        x[i]*=D[i];
    }
    return (0);
}
int itsol_preconILU(double *x, double *y, ITS_PC *mat)
{
    /*-------------------- precon for csr format using the ITS_PC struct*/
    return itsol_lusolC(x, y, mat->ILU);
}
int itsol_pc_assemble(ITS_SOLVER *s)//The initialization of precondition
{
    ITS_PC_TYPE pctype;
    int ierr;
    ITS_PARS p;
    ITS_PC *pc;
    assert(s != NULL);
    pc = &s->pc;

    /* type */
    pctype = pc->pc_type;
    p = s->pars;
    if (pctype == ITS_PC_ILUK) {
        ierr = itsol_pc_ilukC(p.iluk_level, s->csmat_new, pc->ILU, pc->log);//or s->csmat_new

        if (ierr != 0) {
            fprintf(pc->log, "pc assemble, ILUK error\n");
            return ierr;
        }

        pc->precon = itsol_preconILU;
    }
    else {
        fprintf(pc->log, "wrong preconditioner type\n");
        exit(-1);
    }

    return 0;
}
void mv(int n,int *Rowptr,int *ColIndex,double *Value,double *x,double *y)
{
    for(int i=0;i<n;i++)
    {
        y[i]=0.0;
        for(int j=Rowptr[i];j<Rowptr[i+1];j++)
        {
            int k=ColIndex[j];
            y[i]+=Value[j]*x[k];
        }
    }
}
int itsol_solver_assemble(ITS_SOLVER *s,int *RowPtr,int *ColIdx,double *Val,int m,int nnzR)//The intilization of solver
{
    ITS_PC_TYPE pctype;
    ITS_CooMat A;
    int ierr;
    FILE *log;

    assert(s != NULL);

    if (s->assembled) return 0;

    /* log */
    if (s->log == NULL) {
        log = stdout;
    }
    else {
        log = s->log;
    }

    /* assemble */
    pctype = s->pc_type;

    s->csmat = (ITS_SparMat *) itsol_malloc(sizeof(ITS_SparMat), "solver assemble");
    s->csmat_new=(ITS_SparMat *) itsol_malloc(sizeof(ITS_SparMat), "solver assemble");
    //A = *s->A;


    s->csmat_new->RowPtr=(int *)malloc((m+1)*sizeof (int));
    s->csmat_new->ColIdx=(int *)malloc(nnzR*sizeof(int));
    s->csmat_new->Val=(double *)malloc(nnzR*sizeof(double));
    s->csmat_new->n=m;
    for(int i=0;i<m+1;i++)
        s->csmat_new->RowPtr[i]=RowPtr[i];
    for(int i=0;i<m;i++)
    {
        for(int j=s->csmat_new->RowPtr[i];j<s->csmat_new->RowPtr[i+1];j++)
        {
            s->csmat_new->ColIdx[j]=ColIdx[j];
            s->csmat_new->Val[j]=Val[j];
        }
    }
    printf("m=%d\n",m);
    s->smat.CS=s->csmat_new;
    s->smat.n=m;
    /* pc assemble */
    itsol_pc_assemble(s);//����Ԥ����

    s->assembled = 1;
    return 0;
}
int itsol_solver_bicgstab(ITS_SMat *Amat, ITS_PC *lu, double *rhs, double *x, ITS_PARS io,
        int *nits, double *res)//bicg
{
    double *rg, *rh, *pg, *ph, *sg, *sh, *tg, *vg, *tp;
    double r0 = 0, r1 = 0, pra = 0, prb = 0, prc = 0;
    double residual, err_rel = 0;
    int i, n, retval = 0;
    int itr = 0.;
    double tol = io.tol;
    int maxits = io.maxits;
    FILE * fp = io.fp;
    ITS_SparMat *CS=Amat->CS;
    n = Amat->n;
    spmv_Handle_t balanced_handle;
    int nthreads=8;//线程数
    spmv_create_handle_all_in_one(&balanced_handle, n, CS->RowPtr, CS->ColIdx, CS->Val, nthreads,
                                  Method_Balanced2, sizeof(CS->Val[0] ), VECTOR_NONE);
    rg=(double *)malloc(n*sizeof(double));
    //rg = itsol_malloc(n * sizeof(double), "bicgstab");
    rh = (double *)malloc(n*sizeof(double));
    pg = (double *)malloc(n*sizeof(double));
    ph = (double *)malloc(n*sizeof(double));
    sg = (double *)malloc(n*sizeof(double));
    sh = (double *)malloc(n*sizeof(double));
    tg = (double *)malloc(n*sizeof(double));
    vg = (double *)malloc(n*sizeof(double));
    tp = (double *)malloc(n*sizeof(double));
    //Amat->matvec(Amat, x, tp);
    //parallel_balanced_gemv(balanced_handle,n,CS->RowPtr,CS->ColIdx,CS->Val,x,tp);
    spmv(balanced_handle,n,CS->RowPtr,CS->ColIdx,CS->Val,x,tp);
    //spmv_serial_double_VECTOR_NONE(n,CS->RowPtr,CS->ColIdx,CS->Val,x,tp);
    printf("%d",n);
    //mv(n,CS->RowPtr,CS->ColIdx,CS->Val,x,tp);
    for (i = 0; i < n; i++) rg[i] = rhs[i] - tp[i];

    for (i = 0; i < n; i++) {
        rh[i] = rg[i];
        sh[i] = ph[i] = 0.;
    }

    residual = err_rel = itsol_norm(rg, n);
    tol = residual * fabs(tol);

    if (tol == 0.) goto skip;

    for (itr = 0; itr < maxits; itr++) {
        r1 = itsol_dot(rg, rh, n);

        if (r1 == 0) {
            if (io.verb > 0 && fp != NULL) fprintf(fp, "solver bicgstab failed.\n");
            break;
        }

        if (itr == 0) {

            for (i = 0; i < n; i++)
                pg[i] = rg[i];
        }
        else {
            prb = (r1 * pra) / (r0 * prc);
            for (i = 0; i < n; i++) {
                pg[i] = rg[i] + prb * (pg[i] - prc * vg[i]);
            }
        }

        r0 = r1;

        /*  pc */
        if (lu == NULL) {
            memcpy(ph, pg, n * sizeof(double));
        }
        else {
            lu->precon(pg, ph, lu);
        }

        //Amat->matvec(Amat, ph, vg);
        //parallel_balanced_gemv(balanced_handle,n,RowPtr,ColIdx,Val,ph,vg);
        spmv(balanced_handle,n,CS->RowPtr,CS->ColIdx,CS->Val,ph,vg);
        //spmv_serial_double_VECTOR_NONE(n,CS->RowPtr,CS->ColIdx,CS->Val,ph,vg);
        //mv(n,CS->RowPtr,CS->ColIdx,CS->Val,ph,vg);
        pra = r1 / itsol_dot(rh, vg, n);
        for (i = 0; i < n; i++) {
            sg[i] = rg[i] - pra * vg[i];
        }

        if (itsol_norm(sg, n) <= 1e-60) {
            for (i = 0; i < n; i++) {
                x[i] = x[i] + pra * ph[i];
            }

            //Amat->matvec(Amat, x, tp);
            //parallel_balanced_gemv(balanced_handle,n,RowPtr,ColIdx,Val,x,tp);
            spmv(balanced_handle,n,CS->RowPtr,CS->ColIdx,CS->Val,x,tp);
            //mv(n,CS->RowPtr,CS->ColIdx,CS->Val,x,tp);
            for (i = 0; i < n; i++) rg[i] = rhs[i] - tp[i];
            residual = itsol_norm(rg, n);

            break;
        }

        if (lu == NULL) {
            memcpy(sh, sg, n * sizeof(double));
        }
        else {
            lu->precon(sg, sh, lu);
        }

        //Amat->matvec(Amat, sh, tg);
        //parallel_balanced_gemv(balanced_handle,n,RowPtr,ColIdx,Val,sh,tg);
        spmv(balanced_handle,n,CS->RowPtr,CS->ColIdx,CS->Val,sh,tg);
        //mv(n,CS->RowPtr,CS->ColIdx,CS->Val,sh,tg);
        prc = itsol_dot(tg, sg, n) / itsol_dot(tg, tg, n);
        for (i = 0; i < n; i++) {
            x[i] = x[i] + pra * ph[i] + prc * sh[i];
            rg[i] = sg[i] - prc * tg[i];
        }

        residual = itsol_norm(rg, n);

        if (io.verb > 0 && fp != NULL) fprintf(fp, "%8d   %10.2e\n", itr, residual / err_rel);

        if (residual <= tol) break;
    }

    if (itr < maxits) itr += 1;

skip:
    free(rg);
    free(rh);
    free(pg);
    free(ph);
    free(sg);
    free(sh);
    free(tg);
    free(tp);
    free(vg);

    if (itr >= maxits) retval = 1;
    if (nits != NULL) *nits = itr;
    if (res != NULL) *res = residual;

    return retval;
}
int itsol_solver_solve(ITS_SOLVER *s, double *x, double *rhs,int *RowPtr,int *ColIdx,double *Val,int m,int nnzR)
{
    ITS_PARS io;
    ITS_PC_TYPE pctype;
    ITS_SOLVER_TYPE stype;

    assert(s != NULL);
    assert(x != NULL);
    assert(rhs != NULL);

    /* assemble */
    itsol_solver_assemble(s,RowPtr,ColIdx,Val,m,nnzR);//coo��ʽת����CSR��ʽ
    printf("sawd");
    io = s->pars;
    pctype = s->pc_type;
    stype = s->s_type;
    return itsol_solver_bicgstab(&s->smat, &s->pc, rhs, x, io, &s->nits, &s->res);
}
int main(int argc, char **argv)
{
    int n, i;
    ITS_SOLVER s;
    char *filename = argv[1];
    printf ("filename = %s\n", filename);
    int m, n_csr, nnzR, isSymmetric;
    mmio_info(&m, &n_csr, &nnzR, &isSymmetric, filename);
    /* case: COO formats */
    int *RowPtr = (int *) aligned_alloc(ALIGENED_SIZE,(m + 1) * sizeof(int));
    int *ColIdx = (int *) aligned_alloc(ALIGENED_SIZE,nnzR * sizeof(int));
    double *Val = (double *) aligned_alloc(ALIGENED_SIZE, nnzR * sizeof(double));
    mmio_data(RowPtr, ColIdx, Val, filename);
    /* solution vectors */
    double *x1 = (double *)itsol_malloc(m * sizeof(double), "main");
    double *rhs1 = (double *)itsol_malloc(m * sizeof(double), "main");

    for(int i=0;i<m;i++)
    {
        x1[i]=0.0;
        rhs1[i]=i;
    }
    /* init */
    itsol_solver_initialize(&s, ITS_SOLVER_BICGSTAB, ITS_PC_ILUK);
    /* call solver */
    itsol_solver_solve(&s, x1, rhs1,RowPtr,ColIdx,Val,m,nnzR);

    /* get results */
    printf("solver converged in %d steps...\n\n", s.nits);

    /* cleanup */
    //itsol_solver_finalize(&s);

    //itsol_cleanCOO(&A);
    free(x1);
    free(rhs1);
    return 0;
}
