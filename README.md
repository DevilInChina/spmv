[toc]



# gemv

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

See at 

```shell
gemv/lib/
```



## Compile some executable samples

See at

```shell
gemv/bin/
```

and their source code locate in

```shell
gemv/src/samples/
```

# Method call

## double & float

Most of codes support double and float by using 

```c
spmv_methodX(......,sizeof(double/float),VECTORIZED_WAY);
```

We also support the call like

```c
spmv_methodX_type_VECTORIZED_WAY(......);
```

Actually function spmv_methodX_type_VECTORIZED_WAY was generate by macro defines.

## Basic Functions





## VECTORIZED_WAY



### none

All of codes support functions use none vectorized method 

### avx2



### avx512



## Different method

### serial



### omp_parallel



### balanced



### balanced2



### sell_C_Sigma



# Matrix inspect and choose best method to run

