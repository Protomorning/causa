cmake_minimum_required(VERSION 3.16)
project(causa VERSION 0.1.0 LANGUAGES C)

# The library is C11 with no external dependencies. The CUDA backend is
# optional and is only built when the CUDA toolkit is found.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# A single-config generator needs to be told what to build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CAUSA_BUILD_TESTS "Build the C test suite" ON)
option(CAUSA_BUILD_SHARED "Build a shared library as well as a static one" OFF)
option(CAUSA_WITH_CUDA "Build the CUDA backend" OFF)

set(CAUSA_SOURCES
    src/causa_arena.c
    src/causa_tensor.c
    src/causa_graph.c
    src/causa_backend_cpu.c
    src/causa_memory.c
    src/causa_scm.c
    src/causa_io.c
)

# The CUDA backend is one source file that is compiled two ways. With nvcc
# it is the real backend; with the C compiler it is a stub that reports
# that there is no CUDA. The file decides by checking __CUDACC__.
if(CAUSA_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CAUSA_WITH_CUDA was asked for but no CUDA compiler was found")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)

    set_source_files_properties(src/causa_backend_cuda.c PROPERTIES LANGUAGE CUDA)
    list(APPEND CAUSA_SOURCES src/causa_backend_cuda.c)
else()
    list(APPEND CAUSA_SOURCES src/causa_backend_cuda.c)
endif()

add_library(causa_static STATIC ${CAUSA_SOURCES})
target_include_directories(causa_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    # /std:c11 is needed for the flexible array member in the arena block.
    target_compile_options(causa_static PRIVATE /W4 /std:c11 /D_CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(causa_static PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CAUSA_WITH_CUDA)
    target_link_libraries(causa_static PUBLIC cudart cublas cudnn)
endif()

# The name of the library is the same whatever the kind of build it is.
set_target_properties(causa_static PROPERTIES OUTPUT_NAME causa)

if(CAUSA_BUILD_SHARED)
    add_library(causa_shared SHARED ${CAUSA_SOURCES})
    target_include_directories(causa_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(causa_shared PROPERTIES OUTPUT_NAME causa)
    if(NOT MSVC)
        target_compile_options(causa_shared PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(CAUSA_WITH_CUDA)
        target_link_libraries(causa_shared PUBLIC cudart cublas cudnn)
    endif()
endif()

if(CAUSA_BUILD_TESTS)
    enable_testing()

    add_executable(causa_test test/test_causa.c)
    target_link_libraries(causa_test PRIVATE causa_static)
    add_test(NAME causa_test COMMAND causa_test)
endif()

# The Python extension is built by setup.py, which knows where the Python
# headers are. CMake is for the C library.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    message(STATUS "Python: ${Python3_EXECUTABLE} (the extension is built with setup.py)")
endif()

install(TARGETS causa_static
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
