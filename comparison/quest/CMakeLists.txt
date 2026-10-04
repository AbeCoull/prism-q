cmake_minimum_required(VERSION 3.21)
project(quest_runner C CXX)

# QuEST comparator for the cross-simulator comparison harness. Fetches QuEST at
# a pinned tag and builds it as a static library with OpenMP multithreading,
# double precision, and no GPU or MPI, so the comparison is CPU against CPU.
include(FetchContent)
set(QUEST_TAG "v4.3.0" CACHE STRING "QuEST git tag to build against")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(QUEST_FLOAT_PRECISION 2 CACHE STRING "" FORCE)
set(QUEST_ENABLE_OMP ON CACHE BOOL "" FORCE)
set(QUEST_ENABLE_MPI OFF CACHE BOOL "" FORCE)
set(QUEST_ENABLE_NUMA OFF CACHE BOOL "" FORCE)
set(QUEST_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(QUEST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(QUEST_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  quest
  GIT_REPOSITORY https://github.com/QuEST-Kit/QuEST.git
  GIT_TAG ${QUEST_TAG}
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(quest)

add_executable(quest_runner main.cpp)
target_compile_features(quest_runner PRIVATE cxx_std_17)
target_compile_definitions(quest_runner PRIVATE QUEST_RUNNER_TAG="${QUEST_TAG}")
find_package(OpenMP REQUIRED)
target_link_libraries(quest_runner PRIVATE QuEST::QuEST OpenMP::OpenMP_CXX)
