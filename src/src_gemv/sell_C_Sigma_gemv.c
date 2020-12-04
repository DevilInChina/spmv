//
// Created by kouushou on 2020/12/4.
//

void sell_C_Sigma_gemv(){
/*
    memset (Y_first, 0, sizeof(float) * (BloS * S));
    memset (Y, 0, sizeof(float) * m);
    int loca = 0;
    for(int k = 0;k < S/C*BloS;k++)
    {
        if(k !=0 )
        {
            loca += Cmax[k-1];
        }
        for(int i = 0;i < C;i++)
        {
            float sum = 0;
            for(int p = 0;p < Cmax[k];p++)
            {
                sum += ValAT[C*loca+p*C+i] * X[ColAT[C*loca+p*C+i]];
            }
            Y_first[k*C+i] += sum;
        }
    }
    for(int i = 0;i < BloS * S;++i)
    {
        Y[RowSort[i]] = Y_first[i];
    }
    if(redBloS != 0)
    {
        for (int i = 0; i < nnzred; i++)
        {
            float sum = 0;
            sum += Valred[i] * X[Colred[i]];
            Y[Rowred[i]] += sum;
        }
    }
    */
}