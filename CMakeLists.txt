cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME cuda_affine)
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output_data)

file(GLOB CUDA_SRC "*.cu")
file(GLOB HOST_SRC "*.cpp" "*.c")
file(GLOB DATA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/data/*.raw")

find_package(CUDA COMPONENTS cudart REQUIRED)
find_package(Boost COMPONENTS program_options timer system REQUIRED)

add_definitions(-std=c++11)
include_directories("./" ${CUDA_INCLUDE_DIRS})

message("CUDA_INCLUDE_DIR - ${CUDA_INCLUDE_DIRS}")
message("Input image file - ${DATA_SRC}")

cuda_add_executable(${PROJECT_NAME} ${CUDA_SRC} ${HOST_SRC})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${Boost_LIBRARIES})

########################################################################
# Create run.sh
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/run.sh
	COMMAND echo ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME} --image ${DATA_SRC} --output ${OUTPUT_DIR}/output_image.raw >> ${CMAKE_CURRENT_BINARY_DIR}/run.sh
	COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/run.sh;
		)
add_custom_target(run_script ALL
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/run.sh)
add_custom_target(build-time-make-directory ALL
	COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR})

