cmake_minimum_required(VERSION 3.10)
project(whisper_diarize)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent
include(FetchContent)

# Fetch cxxopts
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
)
FetchContent_MakeAvailable(cxxopts)

# Find dependencies
find_package(Threads REQUIRED)

# Add whisper.cpp as a subdirectory
add_subdirectory(whisper.cpp)

# Add our source files
add_executable(whisper_diarize
    src/main.cpp
    src/whisper_diarize.cpp
)

target_include_directories(whisper_diarize PRIVATE
    include
    whisper.cpp
    ${cxxopts_SOURCE_DIR}/include
    .
)

target_link_libraries(whisper_diarize PRIVATE
    whisper
    Threads::Threads
    cxxopts::cxxopts
)

# Add the benchmark cleanup utility
add_executable(benchmark_cleanup
    src/benchmark_cleanup.cpp
)

target_include_directories(benchmark_cleanup PRIVATE
    include
)

target_link_libraries(benchmark_cleanup PRIVATE
    Threads::Threads
)

# Add benchmark cleanup utility
add_executable(cleanup_benchmarks
    src/cleanup_benchmarks.cpp
)

# Add benchmark management tool
add_executable(manage_benchmarks
    src/manage_benchmarks.cpp
)

# Add benchmark test program
add_executable(test_benchmark
    src/test_benchmark.cpp
)

target_include_directories(test_benchmark PRIVATE
    include
)

target_link_libraries(test_benchmark PRIVATE
    Threads::Threads
) 