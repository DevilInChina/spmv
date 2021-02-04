cmake_minimum_required(VERSION 3.3)
project(spmv C CXX)
set(CMAKE_CXX_COMPILER g++)
#set(CMAKE_C_COMPILER icc)
set(CMAKE_C_STANDARD 11)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} ")

include_directories(include)

add_compile_options(-mavx -mavx2 -march=native -O3)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()
link_libraries(m)
link_libraries(pthread)
link_libraries(metis)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)       #设置可执行文件的输出目录

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)           #设置库文件的输出目录


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/src_spmv SPMVS)

add_library(mv_l2 SHARED ${SPMVS} )
target_include_directories(mv_l2 PRIVATE src/csr5_avx2)
target_include_directories(mv_l2 PRIVATE src/csr5_avx2/avx2)

#target_compile_definitions(mv PRIVATE OPT_LEVEL=3 )


add_executable(spmv_l2 ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_link_libraries(spmv_l2 mv_l2)


add_library(mv_l3 SHARED ${SPMVS} )
target_include_directories(mv_l3 PRIVATE src/csr5_avx2)
target_include_directories(mv_l3 PRIVATE src/csr5_avx2/avx2)

target_compile_definitions(mv_l3 PRIVATE OPT_LEVEL=3 )


add_executable(spmv_l3 ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_link_libraries(spmv_l3 mv_l3)


option(BUILD_MKL_BENCH "build mkl bench" ON)

if (BUILD_MKL_BENCH)
    message(${MKLROOT})
    add_executable(mkl_spmv ${PROJECT_SOURCE_DIR}/src/samples/mkl_spmv.c)
    link_directories(mkl_spmv PRIVATE ${MKLROOT}/lib/intel64)
    target_include_directories(mkl_spmv PRIVATE ${MKLROOT}/include)
    target_link_libraries(mkl_spmv PRIVATE mkl_rt)

endif ()
#[[
add_executable(compare ${PROJECT_SOURCE_DIR}/src/samples/compare.c)
target_link_libraries(compare mv)

add_executable(temp ${PROJECT_SOURCE_DIR}/src/samples/temp.c)
target_compile_definitions(temp PRIVATE VALUE_TYPE=float)
target_link_libraries(temp mv)

add_executable(serial_spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_compile_options(serial_spmv PRIVATE -D TEST_METHOD=0)
target_link_libraries(serial_spmv mv)


add_executable(parallel_spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_compile_options(parallel_spmv PRIVATE -D TEST_METHOD=1)
target_link_libraries(parallel_spmv mv)


add_executable(balanced_spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_compile_options(balanced_spmv PRIVATE -D TEST_METHOD=2)
target_link_libraries(balanced_spmv mv)


add_executable(2balanced_spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_compile_options(2balanced_spmv PRIVATE -D TEST_METHOD=3)
target_link_libraries(2balanced_spmv mv)


add_executable(sellCSigma_spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_compile_options(sellCSigma_spmv PRIVATE -D TEST_METHOD=4)
target_link_libraries(sellCSigma_spmv mv)


add_executable(csr5_spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)
target_compile_options(csr5_spmv PRIVATE -D TEST_METHOD=5)
target_link_libraries(csr5_spmv mv)
#]]