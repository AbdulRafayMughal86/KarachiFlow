cmake_minimum_required(VERSION 3.20)
project(KarachiFlow VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4 /MP)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

find_package(OpenMP REQUIRED)
find_package(MPI QUIET)

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/engine)

# Network binary reader and benchmark target
if(EXISTS ${CMAKE_SOURCE_DIR}/src/engine/test_network_reader.cpp)
    add_executable(test_network_reader ${CMAKE_SOURCE_DIR}/src/engine/test_network_reader.cpp)
    target_link_libraries(test_network_reader PRIVATE OpenMP::OpenMP_CXX)
endif()

# Main simulation engine target
if(EXISTS ${CMAKE_SOURCE_DIR}/src/main.cpp)
    file(GLOB_RECURSE CORE_SOURCES
        ${CMAKE_SOURCE_DIR}/src/engine/*.cpp
        ${CMAKE_SOURCE_DIR}/src/parallel/*.cpp
    )
    list(FILTER CORE_SOURCES EXCLUDE REGEX ".*test_.*\\.cpp$")
    add_executable(karachiflow ${CMAKE_SOURCE_DIR}/src/main.cpp ${CORE_SOURCES})
    target_link_libraries(karachiflow PRIVATE OpenMP::OpenMP_CXX)
    if(MPI_FOUND)
        target_link_libraries(karachiflow PRIVATE MPI::MPI_CXX)
        target_compile_definitions(karachiflow PRIVATE HAS_MPI=1)
    endif()

    if(EXISTS ${CMAKE_SOURCE_DIR}/src/engine/test_nasch_validation.cpp)
        add_executable(test_nasch_validation ${CMAKE_SOURCE_DIR}/src/engine/test_nasch_validation.cpp ${CORE_SOURCES})
        target_link_libraries(test_nasch_validation PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(EXISTS ${CMAKE_SOURCE_DIR}/src/engine/test_simulation.cpp)
        add_executable(test_simulation ${CMAKE_SOURCE_DIR}/src/engine/test_simulation.cpp ${CORE_SOURCES})
        target_link_libraries(test_simulation PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

