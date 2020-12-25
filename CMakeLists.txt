cmake_minimum_required(VERSION 3.16)
project(spmv C CXX)
set(CMAKE_CXX_COMPILER g++)
#set(CMAKE_C_COMPILER icc)
set(CMAKE_C_STANDARD 11)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} ")

include_directories(include)

add_compile_options(-mavx -mavx2 -march=native )

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()
link_libraries(m)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)       #设置可执行文件的输出目录

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)           #设置库文件的输出目录


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/src_spmv SPMVS)

add_library(mv SHARED ${SPMVS} )
target_include_directories(mv PRIVATE src/csr5_avx2)
target_include_directories(mv PRIVATE src/csr5_avx2/avx2)

target_compile_definitions(mv PRIVATE DOT_AVX2_CAN )
#target_compile_definitions(mv PRIVATE DOT_AVX512_CAN )

add_executable(cg ${PROJECT_SOURCE_DIR}/src/samples/cg.c)

target_link_libraries(cg mv)

add_executable(bicg ${PROJECT_SOURCE_DIR}/src/samples/bicg.c)

target_link_libraries(bicg mv)

add_executable(spmv ${PROJECT_SOURCE_DIR}/src/samples/test_spmv.c)

target_link_libraries(spmv mv)

add_executable(line ${PROJECT_SOURCE_DIR}/src/samples/test_lineProduct.c)

target_link_libraries(line mv)
add_executable(pagerank ${PROJECT_SOURCE_DIR}/src/samples/PageRank.c)

target_link_libraries(pagerank mv)
