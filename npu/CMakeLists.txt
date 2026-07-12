# CMakeLists.txt — NPU age/gender detector (age_gender_npu).
#
# Links the working qnn_py::QnnRuntime (a thin wrapper over the QNN SampleApp,
# patched to expose runtime-only load + host-buffer execute) straight into a
# C/C++ app, with no Python in the loop. Builds on the board only (ARM64 QNN).
cmake_minimum_required(VERSION 3.18)
project(age_gender_npu LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- QAIRT SDK + the working qnn_py runtime (both live in the user's home) ---
if(NOT DEFINED QAIRT_ROOT)
  set(QAIRT_ROOT "$ENV{HOME}/qairt/2.42.0.251225")
endif()
if(NOT DEFINED QNN_RUNTIME_DIR)
  set(QNN_RUNTIME_DIR "${QAIRT_ROOT}/qnn_step2_step3_real_runtime_patch/qnn_py_runtime")
endif()

set(SAMPLEAPP_SRC "${QAIRT_ROOT}/examples/QNN/SampleApp/SampleApp/src")
if(NOT EXISTS "${SAMPLEAPP_SRC}/QnnSampleApp.cpp")
  message(FATAL_ERROR "QnnSampleApp.cpp not found at ${SAMPLEAPP_SRC}")
endif()
if(NOT EXISTS "${QNN_RUNTIME_DIR}/src/QnnRuntime.cpp")
  message(FATAL_ERROR "QnnRuntime.cpp not found at ${QNN_RUNTIME_DIR}/src")
endif()

# --- QNN SampleApp sources (same set the pybind module compiles, minus main) -
file(GLOB SAMPLE_ROOT_CPP    CONFIGURE_DEPENDS "${SAMPLEAPP_SRC}/*.cpp")
file(GLOB SAMPLE_UTILS_CPP   CONFIGURE_DEPENDS "${SAMPLEAPP_SRC}/Utils/*.cpp")
file(GLOB SAMPLE_WRAPPER_CPP CONFIGURE_DEPENDS "${SAMPLEAPP_SRC}/WrapperUtils/*.cpp")
file(GLOB SAMPLE_LOG_CPP     CONFIGURE_DEPENDS "${SAMPLEAPP_SRC}/Log/*.cpp")
file(GLOB SAMPLE_PAL_COMMON  CONFIGURE_DEPENDS "${SAMPLEAPP_SRC}/PAL/src/common/*.cpp")
file(GLOB SAMPLE_PAL_LINUX   CONFIGURE_DEPENDS "${SAMPLEAPP_SRC}/PAL/src/linux/*.cpp")
list(REMOVE_ITEM SAMPLE_ROOT_CPP "${SAMPLEAPP_SRC}/main.cpp")

add_library(qnn_runtime STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/qnn/QnnRuntime.cpp
  ${SAMPLE_ROOT_CPP}
  ${SAMPLE_UTILS_CPP}
  ${SAMPLE_WRAPPER_CPP}
  ${SAMPLE_LOG_CPP}
  ${SAMPLE_PAL_COMMON}
  ${SAMPLE_PAL_LINUX}
)
target_include_directories(qnn_runtime PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${QAIRT_ROOT}/include"
  "${QAIRT_ROOT}/include/QNN"
  "${SAMPLEAPP_SRC}"
  "${SAMPLEAPP_SRC}/Utils"
  "${SAMPLEAPP_SRC}/WrapperUtils"
  "${SAMPLEAPP_SRC}/Log"
  "${SAMPLEAPP_SRC}/PAL/include"
)
target_compile_options(qnn_runtime PRIVATE
  -O3 -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(qnn_runtime PUBLIC dl pthread)

# --- the app -----------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(age_gender_npu
  main_npu.c
  app_npu.cpp
)
target_include_directories(age_gender_npu PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(age_gender_npu PRIVATE
  qnn_runtime ${OpenCV_LIBS} Threads::Threads dl)

# QNN backends (libQnnHtp.so + V68 stub) are dlopen'ed by absolute path at
# runtime; keep the SDK lib dirs on the rpath as a fallback.
set_target_properties(age_gender_npu PROPERTIES
  BUILD_RPATH "${QAIRT_ROOT}/lib/aarch64-oe-linux-gcc11.2;${QAIRT_ROOT}/lib/aarch64-ubuntu-gcc9.4")
