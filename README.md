[toc]



# spmv

some methods to calculate Matrix product vector.

including serial , omp_parallel , balanced , balanced2 , sell_C_Sigma

# Build

## Use cmake and shell commands to compile this project

```shell
mkdir build
cd build
cmake ..
make
#and then get executable file at ../bin and libs at ../lib
```

## Compile to a shared library

### Lib location

See at 

```shell
spmv/lib/
```
### Defines
#### AVX2 & AVX512
see at [Performance avx2 and avx512](#Performance avx2 and avx512)



## Compile some executable samples

### Bin location

```shell
spmv/bin/
```

### Source code location

```shell
spmv/src/samples/
```

# Method call
```c
/// declaration in spmv.h

/**
 * @brief initialize a handle according to parameters sent in ,
 * @param Handle
 * @param m                 rows of the csr-storage matrix
 * @param RowPtr            length is (m+1) , RowPtr[i]-RowPtr[i-1] means the number of non-zero element at line i
 * @param ColIdx            length is (RowPtr[m]-RowPtr[0]) , means Collum index of each non-zero element
 * @param Matrix_Val        (void*) length is (RowPtr[m]-RowPtr[0]) , means Value of each non-zero element
 * @param nthreads          max number of threads can be use
 * @param Function          calculate way (serial,parallel,parallel_balanced,parallel_balanced2,sell_C_Sigma)
 * @param size              sizeof(double)/sizeof(float) refer to call float or double version
 * @param vectorizedWay     using (not use,avx2,avx512)
 */
void spmv_create_handle_all_in_one(
            spmv_Handle_t *Handle,
            BASIC_INT_TYPE m,
            const BASIC_INT_TYPE *RowPtr,
            const BASIC_INT_TYPE *ColIdx,
            const void *Matrix_Val,
            BASIC_SIZE_TYPE nthreads,
            SPMV_METHODS Function,
            BASIC_SIZE_TYPE size,
            VECTORIZED_WAY vectorizedWay
);


/**
 * @brief calculate according to handle
 * @param handle            must call function spmv_create_handle_all_in_one before
 * @param m                 rows of the csr-storage matrix
 * @param RowPtr            length is (m+1) , RowPtr[i]-RowPtr[i-1] means the number of non-zero element at line i
 * @param ColIdx            length is (RowPtr[m]-RowPtr[0]) , means Collum index of each non-zero element
 * @param Matrix_Val        (void*) length is (RowPtr[m]-RowPtr[0]) , means Value of each non-zero element
 * @param Vector_Val_X      length is n or max(ColIdx) (max element in ColIdx)
 * @param Vector_Val_Y      length is m
 */
void spmv(
            const spmv_Handle_t handle,
            BASIC_INT_TYPE m,
            const BASIC_INT_TYPE* RowPtr,
            const BASIC_INT_TYPE* ColIdx,
            const void* Matrix_Val,
            const void* Vector_Val_X,
            void*       Vector_Val_Y
);



```

## double & float

```c
spmv_Handle_t new_handle = NULL;

spmv_create_handle_all_in_one(&new_handle,m,RowPtr,ColIdx,Matrix_Val,nthreads,Method_X
    sizeof(double)/*sizeof(float)*/,
    VECTOR_X
);

spmv(new_handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y);

```

## Basic Functions
```c
///spmv_Defines.h
typedef enum SPMV_METHODS{
    Method_Serial,
    Method_Parallel,
    Method_Balanced,
    Method_Balanced2,
    Method_SellCSigma,
    Method_Total_Size /// count total ways of methods
}SPMV_METHODS;

spmv_Handle_t new_handle = NULL;

spmv_create_handle_all_in_one(&new_handle,m,RowPtr,ColIdx,Matrix_Val,nthreads,
    Method_X,
    sizeof(X),VECTOR_X
);

spmv(new_handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y);

```


## VECTORIZED_WAY
```c
///spmv_Defines.h
typedef enum VECTORIZED_WAY{
VECTOR_NONE,
VECTOR_AVX2,
VECTOR_AVX512,
VECTOR_TOTAL_SIZE/// count total ways of vectorized
}VECTORIZED_WAY;

spmv_Handle_t new_handle = NULL;

spmv_create_handle_all_in_one(&new_handle,m,RowPtr,ColIdx,Matrix_Val,nthreads,Method_X,sizeof(X),
    VECTOR_X
);

spmv(new_handle,m,RowPtr,ColIdx,Matrix_Val,Vector_Val_X,Vector_Val_Y);

```
### Performance avx2 and avx512

add to CMakeLists.txt after

```cmake
add_library(mv SHARED ${SPMVS})
```



```cmake
target_compile_definitions(mv PRIVATE DOT_AVX2_CAN)

target_compile_definitions(mv PRIVATE DOT_AVX512_CAN)
```

and recompile lib to make work properly

#### none

All of codes support functions use none vectorized method 

#### avx2

If defined DOT_AVX2_CAN program will use AVX2 to calculate dot product and line product

Otherwise use none to calculate . 

#### avx512

If defined DOT_AVX512_CAN program will use AVX2 to calculate dot product and line product

Otherwise use AVX2 to calculate . 

## Different method

### serial

Just use dot product and choose vectorized way 

### omp_parallel

Just use omp parallel

### balanced

Simple add load balancing to speed up processing , as a result need to use 'spmv_create_handle_all_in_one' to cost some time to create a handle.

### balanced2

Deeper load balancing than balanced , need more time to prepare.

### sell_C_Sigma



# Matrix inspect and choose best method to run

