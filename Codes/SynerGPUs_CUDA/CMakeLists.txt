cmake_minimum_required(3.26.4)
project(SynerGPUs_CUDA)

file(GLOB SRC_FILES
    "src/*.cpp"
)

set(INC_FILES_DIR
    include/
)

include_directories(
    ${INC_FILES_DIR}
)

add_executable(SynerGPUs_CUDA
    ${SRC_FILES}
)
