cmake_minimum_required(VERSION 3.16)
project(add_lora_swiglu_quant_standalone CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910b4")
endif()
set(ASCEND_COMPUTE_UNIT ${SOC_VERSION})

message(STATUS "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
message(STATUS "SOC_VERSION=${SOC_VERSION}")

if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_HOME_PATH}")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(swiglu_quant_kernel SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/add_lora_swiglu_quant.cpp
)

add_executable(bench ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_directories(bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(bench PRIVATE
    swiglu_quant_kernel
    ascendcl
)
