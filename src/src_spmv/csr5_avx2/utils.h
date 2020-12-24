#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>

double getB(const int m, const int nnz, int type)
{
    return 1.0*
    ((m + 1 + nnz) *
    4 + (2 * nnz + m)
    * type);
}

template<typename iT>
double getFLOP(const iT nnz)
{
    return (double)(2 * nnz);
}

template<typename T>
void print_tile_t(T *input, int m, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int local_id = 0; local_id < m; local_id++)
        {
            cout << input[local_id * n + i] << ", ";
        }
        cout << endl;
    }
}

template<typename T>
void print_tile(T *input, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int local_id = 0; local_id < n; local_id++)
        {
            cout << input[i * n + local_id] << ", ";
        }
        cout << endl;
    }
}

template<typename T>
void print_1darray(T *input, int l)
{
    for (int i = 0; i < l; i++)
        cout << input[i] << ", ";
    cout << endl;
}

#endif // UTILS_H
