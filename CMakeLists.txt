cmake_minimum_required(VERSION 3.18)

# Check if CUDA is available before enabling
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(NBodySimulation LANGUAGES CXX CUDA)
    set(CUDA_AVAILABLE TRUE)
else()
    project(NBodySimulation LANGUAGES CXX)
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found - GPU acceleration disabled")
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_MPI "Enable MPI distributed computing" ON)
option(ENABLE_GPROF "Enable gprof profiling" OFF)
option(ENABLE_GCOV "Enable gcov code coverage" OFF)
option(ENABLE_LIKWID "Enable LIKWID performance counters" OFF)

# CUDA option - only enable if CUDA is available
if(CUDA_AVAILABLE)
    option(USE_CUDA "Enable CUDA GPU acceleration" ON)
else()
    set(USE_CUDA OFF)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# ============================================================================
# Profiling Configurations
# ============================================================================

# gprof profiling
if(ENABLE_GPROF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    message(STATUS "gprof profiling enabled")
endif()

# gcov code coverage
if(ENABLE_GCOV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    message(STATUS "gcov code coverage enabled")
endif()

# LIKWID performance counters
if(ENABLE_LIKWID)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(LIKWID likwid)
    endif()

    # Fallback if pkg-config didn't find it
    if(NOT LIKWID_FOUND)
        find_path(LIKWID_INCLUDE_DIR likwid.h PATHS /usr/include /usr/local/include)
        find_library(LIKWID_LIBRARY likwid PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
        
        if(LIKWID_INCLUDE_DIR AND LIKWID_LIBRARY)
            set(LIKWID_FOUND TRUE)
            set(LIKWID_INCLUDE_DIRS ${LIKWID_INCLUDE_DIR})
            set(LIKWID_LIBRARIES ${LIKWID_LIBRARY})
        endif()
    endif()

    if(LIKWID_FOUND)
        add_definitions(-DLIKWID_PERFMON)
        include_directories(${LIKWID_INCLUDE_DIRS})
        # Only link directories if we have them from pkg-config
        if(LIKWID_LIBRARY_DIRS)
            link_directories(${LIKWID_LIBRARY_DIRS})
        endif()
        message(STATUS "LIKWID found: ${LIKWID_LIBRARIES}")
    else()
        message(WARNING "LIKWID not found, disabling performance counters")
        set(ENABLE_LIKWID OFF)
    endif()
endif()

# ============================================================================
# Find Required Packages
# ============================================================================

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# MPI
if(USE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_VERSION}")
        include_directories(${MPI_CXX_INCLUDE_DIRS})
    endif()
endif()

# CUDA
if(USE_CUDA AND CUDA_AVAILABLE)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    
    # CUDA architecture (auto-detect or specify)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
else()
    set(USE_CUDA OFF)
endif()

# ============================================================================
# Source Files
# ============================================================================

# Common sources
set(COMMON_SOURCES
    src/common/particle.cpp
    src/common/config.cpp
)

# Serial sources
set(SERIAL_SOURCES
    src/serial/generators.cpp
    src/serial/direct_nbody.cpp
    src/serial/octree.cpp
    src/serial/barnes_hut.cpp
)

# OpenMP sources
set(OPENMP_SOURCES
    src/openmp/direct_omp.cpp
    src/openmp/barnes_hut_omp.cpp
)

# MPI sources
set(MPI_SOURCES
    src/mpi/domain.cpp
    src/mpi/nbody_mpi.cpp
)

# CUDA sources
set(CUDA_SOURCES
    src/cuda/nbody_cuda.cu
)

# ============================================================================
# Include Directories
# ============================================================================

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Main Executable
# ============================================================================

set(ALL_SOURCES
    src/main.cpp
    ${COMMON_SOURCES}
    ${SERIAL_SOURCES}
)

if(USE_OPENMP)
    list(APPEND ALL_SOURCES ${OPENMP_SOURCES})
endif()

if(USE_MPI)
    list(APPEND ALL_SOURCES ${MPI_SOURCES})
endif()

if(USE_CUDA)
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

add_executable(nbody ${ALL_SOURCES})

# Link libraries
if(USE_OPENMP)
    target_link_libraries(nbody OpenMP::OpenMP_CXX)
    target_compile_definitions(nbody PRIVATE USE_OPENMP)
endif()

if(USE_MPI)
    target_link_libraries(nbody MPI::MPI_CXX)
    target_compile_definitions(nbody PRIVATE USE_MPI)
endif()

if(USE_CUDA)
    target_link_libraries(nbody CUDA::cudart)
    target_compile_definitions(nbody PRIVATE USE_CUDA)
    set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(ENABLE_LIKWID AND LIKWID_FOUND)
    target_link_libraries(nbody ${LIKWID_LIBRARIES})
endif()

target_link_libraries(nbody m)

# ============================================================================
# Profiling Builds (separate executables)
# ============================================================================

# gprof version
add_executable(nbody_gprof ${ALL_SOURCES})
target_compile_options(nbody_gprof PRIVATE -pg -O2)
target_link_options(nbody_gprof PRIVATE -pg)

if(USE_OPENMP)
    target_link_libraries(nbody_gprof OpenMP::OpenMP_CXX)
    target_compile_definitions(nbody_gprof PRIVATE USE_OPENMP)
endif()
if(USE_CUDA)
    target_link_libraries(nbody_gprof CUDA::cudart)
    target_compile_definitions(nbody_gprof PRIVATE USE_CUDA)
endif()
target_link_libraries(nbody_gprof m)

# gcov version
add_executable(nbody_gcov ${ALL_SOURCES})
target_compile_options(nbody_gcov PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
target_link_options(nbody_gcov PRIVATE --coverage)

if(USE_OPENMP)
    target_link_libraries(nbody_gcov OpenMP::OpenMP_CXX)
    target_compile_definitions(nbody_gcov PRIVATE USE_OPENMP)
endif()
if(USE_CUDA)
    target_link_libraries(nbody_gcov CUDA::cudart)
    target_compile_definitions(nbody_gcov PRIVATE USE_CUDA)
endif()
target_link_libraries(nbody_gcov m)

# ============================================================================
# Tests
# ============================================================================

enable_testing()

add_executable(test_correctness tests/test_correctness.cpp ${COMMON_SOURCES} ${SERIAL_SOURCES})
target_link_libraries(test_correctness m)

add_test(NAME CorrectnessTest COMMAND test_correctness)

# ============================================================================
# Separate Executables for Each Mode
# ============================================================================

# Serial executable
add_executable(nbody_serial 
    src/main_serial.cpp 
    ${COMMON_SOURCES} 
    ${SERIAL_SOURCES}
)
target_link_libraries(nbody_serial m)

# OpenMP executable
if(USE_OPENMP)
    add_executable(nbody_openmp 
        src/main_openmp.cpp 
        ${COMMON_SOURCES} 
        ${SERIAL_SOURCES}
        ${OPENMP_SOURCES}
    )
    target_link_libraries(nbody_openmp OpenMP::OpenMP_CXX m)
endif()

# MPI executable
if(USE_MPI)
    add_executable(nbody_mpi 
        src/main_mpi.cpp 
        ${COMMON_SOURCES} 
        ${SERIAL_SOURCES}
        ${MPI_SOURCES}
    )
    target_link_libraries(nbody_mpi MPI::MPI_CXX m)
endif()

# CUDA executable
if(USE_CUDA)
    add_executable(nbody_cuda 
        src/main_cuda.cpp 
        ${COMMON_SOURCES} 
        ${SERIAL_SOURCES}
        ${CUDA_SOURCES}
    )
    target_link_libraries(nbody_cuda CUDA::cudart m)
    set_target_properties(nbody_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ============================================================================
# Install
# ============================================================================

install(TARGETS nbody DESTINATION bin)
install(DIRECTORY scripts/ DESTINATION share/nbody/scripts)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP: ${USE_OPENMP}")
message(STATUS "MPI: ${USE_MPI}")
message(STATUS "CUDA: ${USE_CUDA}")
message(STATUS "gprof: ${ENABLE_GPROF}")
message(STATUS "gcov: ${ENABLE_GCOV}")
message(STATUS "LIKWID: ${ENABLE_LIKWID}")
message(STATUS "===========================")
message(STATUS "")
